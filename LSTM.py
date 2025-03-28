import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        Khởi tạo dataset

        Args:
            data (numpy.ndarray): Dữ liệu đã được chuẩn hóa
            seq_length (int): Độ dài chuỗi đầu vào (số ngày lịch sử để dự đoán)
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # Lấy dữ liệu chuỗi X (đầu vào)
        x = self.data[index:index + self.seq_length]
        # Lấy giá trị mục tiêu y (ngày tiếp theo)
        y = self.data[index + self.seq_length]

        # Chuyển đổi thành tensor
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        return x, y


class LSTMModel(nn.Module):
    """
    Mô hình LSTM để dự đoán giá cổ phiếu
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        """
        Khởi tạo mô hình LSTM

        Args:
            input_size (int): Số feature đầu vào (thường là 1 nếu chỉ dùng giá đóng cửa)
            hidden_size (int): Số đơn vị ẩn trong LSTM
            num_layers (int): Số lớp LSTM xếp chồng
            output_size (int): Số feature đầu ra (thường là 1 cho dự đoán giá)
            dropout_rate (float): Tỷ lệ dropout để tránh overfitting
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Lớp LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Lớp dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Lớp fully connected
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass của mô hình

        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Giá trị dự đoán
        """
        batch_size = x.size(0)

        # Khởi tạo trạng thái ẩn
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Lấy đầu ra từ bước thời gian cuối cùng
        out = out[:, -1, :]

        # Áp dụng dropout
        out = self.dropout(out)

        # Đưa qua lớp fully connected
        out = self.fc(out)

        return out


class StockPricePredictor:
    """
    Class quản lý quá trình tiền xử lý dữ liệu, huấn luyện và dự đoán
    """

    def __init__(self, seq_length=60, hidden_size=50, num_layers=2, learning_rate=0.001, batch_size=64):
        """
        Khởi tạo bộ dự đoán giá cổ phiếu

        Args:
            seq_length (int): Số ngày lịch sử để dự đoán ngày tiếp theo
            hidden_size (int): Số đơn vị ẩn trong LSTM
            num_layers (int): Số lớp LSTM
            learning_rate (float): Tốc độ học
            batch_size (int): Kích thước batch
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Mô hình sẽ được khởi tạo sau khi có dữ liệu
        self.model = None
        self.input_size = None
        self.output_size = None

    def preprocess_data(self, df, target_col='close_price'):
        """
        Tiền xử lý dữ liệu

        Args:
            df (pandas.DataFrame): DataFrame chứa dữ liệu cổ phiếu
            target_col (str): Tên cột chứa giá cổ phiếu muốn dự đoán

        Returns:
            tuple: Bộ dữ liệu huấn luyện và kiểm thử
        """
        # Trích xuất feature (ở đây chỉ dùng giá đóng cửa)
        data = df[[target_col]].values

        # Chuẩn hóa dữ liệu
        scaled_data = self.scaler.fit_transform(data)

        # Xác định kích thước train/test (80% train, 20% test)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.seq_length:]

        # Tạo dataset
        train_dataset = StockDataset(train_data, self.seq_length)
        test_dataset = StockDataset(test_data, self.seq_length)

        # Tạo dataloader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Lưu trữ kích thước đầu vào và đầu ra
        self.input_size = 1  # Chỉ sử dụng một feature (giá đóng cửa)
        self.output_size = 1  # Dự đoán một giá trị (giá đóng cửa)

        # Khởi tạo mô hình
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        ).to(self.device)

        return train_loader, test_loader

    def train(self, train_loader, num_epochs=50):
        """
        Huấn luyện mô hình

        Args:
            train_loader (DataLoader): DataLoader cho dữ liệu huấn luyện
            num_epochs (int): Số epoch huấn luyện
        """
        # Định nghĩa hàm mất mát và bộ tối ưu hóa
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Theo dõi quá trình huấn luyện
        train_losses = []

        # Bắt đầu huấn luyện
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                # Chuyển dữ liệu đến device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Đảm bảo kích thước đúng
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], self.input_size)

                # Forward pass
                y_pred = self.model(X_batch)

                # Tính loss
                loss = criterion(y_pred, y_batch)

                # Backward pass và tối ưu hóa
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)

            # In thông tin huấn luyện
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')

        return train_losses

    def evaluate(self, test_loader):
        """
        Đánh giá mô hình trên tập kiểm thử

        Args:
            test_loader (DataLoader): DataLoader cho dữ liệu kiểm thử

        Returns:
            tuple: Loss trên tập kiểm thử và dự đoán
        """
        criterion = nn.MSELoss()
        self.model.eval()
        test_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # Chuyển dữ liệu đến device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Đảm bảo kích thước đúng
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], self.input_size)

                # Forward pass
                y_pred = self.model(X_batch)

                # Tính loss
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()

                # Lưu kết quả
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        # Chuyển dự đoán về giá thực
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        actuals = self.scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

        avg_test_loss = test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss:.6f}')

        return avg_test_loss, predictions, actuals

    def predict_next_day(self, last_sequence):
        """
        Dự đoán giá cổ phiếu của ngày tiếp theo

        Args:
            last_sequence (numpy.ndarray): Dữ liệu chuỗi thời gian cuối cùng

        Returns:
            float: Giá cổ phiếu dự đoán
        """
        # Chuẩn hóa dữ liệu
        scaled_data = self.scaler.transform(last_sequence.reshape(-1, 1))

        # Chuyển đổi thành tensor
        X = torch.FloatTensor(scaled_data).reshape(1, self.seq_length, self.input_size).to(self.device)

        # Dự đoán
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)

        # Chuyển về giá thực
        predicted_price = self.scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))

        return predicted_price[0][0]

    def save_model(self, path):
        """
        Lưu mô hình

        Args:
            path (str): Đường dẫn để lưu mô hình
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'seq_length': self.seq_length
        }, path)
    def load_model(self, path):
        """
        Tải mô hình

        Args:
            path (str): Đường dẫn đến mô hình đã lưu
        """
        checkpoint = torch.load(path,weights_only=False)

        # Trích xuất thông tin
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.output_size = checkpoint['output_size']
        self.seq_length = checkpoint['seq_length']
        self.scaler = checkpoint['scaler']

        # Tái tạo mô hình
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        ).to(self.device)

        # Tải trạng thái
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


# Ví dụ sử dụng
def example_usage():
    """
    Ví dụ cách sử dụng các class để huấn luyện và dự đoán
    """

    data = pd.read_csv('VN100_stock_price_1D.csv')
    df = data[data['ticker'] == 'HPG']
    # Khởi tạo bộ dự đoán
    predictor = StockPricePredictor(
        seq_length=60,
        hidden_size=50,
        num_layers=2,
        learning_rate=0.001,
        batch_size=64
    )

    # Tiền xử lý dữ liệu
    train_loader, test_loader = predictor.preprocess_data(df, target_col='close_price')

    # Huấn luyện mô hình
    train_losses = predictor.train(train_loader, num_epochs=50)

    # Đánh giá mô hình
    test_loss, predictions, actuals = predictor.evaluate(test_loader)

    # Dự đoán giá cổ phiếu cho ngày tiếp theo
    last_sequence = df['close_price'].values[-60:]
    next_day_price = predictor.predict_next_day(last_sequence)
    print(f'Dự đoán giá cổ phiếu cho ngày tiếp theo: {next_day_price:.2f}')

    # Lưu mô hình
    predictor.save_model('lstm_stock_model.pth')

    # Tải mô hình
    new_predictor = StockPricePredictor()
    new_predictor.load_model('lstm_stock_model.pth')

    # Sử dụng mô hình đã tải để dự đoán
    loaded_prediction = new_predictor.predict_next_day(last_sequence)
    print(f'Dự đoán từ mô hình đã tải: {loaded_prediction:.2f}')


if __name__ == "__main__":
    example_usage()