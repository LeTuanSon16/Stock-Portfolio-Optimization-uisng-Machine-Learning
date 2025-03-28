import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class StockPricePredictor:
    def __init__(self, seq_length=60, hidden_size=50, num_layers=2, learning_rate=0.001, batch_size=64):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Set up device with better error handling and info
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self.device = torch.device('cpu')
                print("No GPU available, using CPU")
        except:
            self.device = torch.device('cpu')
            print("Error detecting GPU, falling back to CPU")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Will be initialized later
        self.model = None
        self.input_size = None
        self.output_size = None
        self.last_date = None
        self.date_column = None

    def preprocess_data(self, df, target_col='close_price', date_col='date'):
        # Save date column and convert to datetime
        self.date_column = date_col
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort and save last date
        df = df.sort_values(by=date_col)
        self.last_date = df[date_col].iloc[-1]
        
        # Extract and normalize data
        data = df[[target_col]].values
        scaled_data = self.scaler.fit_transform(data)
        
        # Split data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.seq_length:]
        
        # Create datasets and dataloaders
        train_dataset = StockDataset(train_data, self.seq_length)
        test_dataset = StockDataset(test_data, self.seq_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Set model dimensions
        self.input_size = 1
        self.output_size = 1
        
        # Initialize model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        ).to(self.device)
        
        return train_loader, test_loader, train_data, test_data

    def calculate_metrics(self, y_true, y_pred):
        """Calculate MSE, RMSE, MPA (Mean Prediction Accuracy)"""
        # Convert to numpy arrays if they are tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
            
        # Reshape if needed
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
            
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean Prediction Accuracy (MPA)
        mpa = 1 - np.mean(np.abs((y_true - y_pred) / y_true))
        
        # Direction Accuracy (correct prediction of up/down movement)
        if len(y_true) > 1:
            actual_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)
            direction_accuracy = np.mean((actual_diff * pred_diff) > 0)
        else:
            direction_accuracy = None
            
        return {
            'mse': mse,
            'rmse': rmse,
            'mpa': mpa,
            'direction_accuracy': direction_accuracy
        }

    def train(self, train_loader, num_epochs=50, use_gpu_metrics=True):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Track training metrics
        train_losses = []
        train_metrics_history = []
        
        # Print GPU info if using CUDA
        if self.device.type == 'cuda':
            print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
            
            # Enable cuDNN benchmark for faster training
            torch.backends.cudnn.benchmark = True
            
            # Optional: Set precision to mixed precision for faster training
            # Requires PyTorch 1.6+ and CUDA
            try:
                from torch.cuda.amp import autocast, GradScaler
                use_amp = True
                scaler = GradScaler()
                print("Using mixed precision training")
            except ImportError:
                use_amp = False
                print("Mixed precision training not available, using full precision")
        else:
            use_amp = False
            print("Training on CPU")
        
        # Show GPU memory usage if available
        if use_gpu_metrics and self.device.type == 'cuda':
            import gc
            from time import time
            start_time = time()
            
            # Initial GPU memory usage
            torch.cuda.empty_cache()
            gc.collect()
            initial_memory = torch.cuda.memory_allocated(0) / 1e9  # GB
            print(f"Initial GPU memory usage: {initial_memory:.3f} GB")
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            all_y_true = []
            all_y_pred = []
            
            for X_batch, y_batch in train_loader:
                # Move to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Reshape input
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], self.input_size)
                
                # Forward pass with potential mixed precision
                if use_amp and self.device.type == 'cuda':
                    with autocast():
                        y_pred = self.model(X_batch)
                        loss = criterion(y_pred, y_batch)
                    
                    # Scale gradients and optimize
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Traditional forward pass
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Accumulate loss and predictions for metrics
                epoch_loss += loss.item()
                all_y_true.append(y_batch.detach().cpu().numpy())
                all_y_pred.append(y_pred.detach().cpu().numpy())
            
            # Calculate average loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            
            # Calculate training metrics for this epoch
            all_y_true = np.concatenate(all_y_true)
            all_y_pred = np.concatenate(all_y_pred)
            
            # Convert normalized values back to original scale for metrics
            y_true_original = self.scaler.inverse_transform(all_y_true)
            y_pred_original = self.scaler.inverse_transform(all_y_pred)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true_original, y_pred_original)
            train_metrics_history.append(metrics)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}, '
                      f'MPA: {metrics["mpa"]:.4f}, Direction Accuracy: {metrics["direction_accuracy"]:.4f}')
        
        # Final GPU memory usage stats
        if use_gpu_metrics and self.device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated(0) / 1e9  # GB
            total_time = time() - start_time
            print(f"\nGPU Training Stats:")
            print(f"- Final GPU memory usage: {final_memory:.3f} GB")
            print(f"- Memory increase: {final_memory - initial_memory:.3f} GB")
            print(f"- Total training time: {total_time:.2f} seconds")
            print(f"- Time per epoch: {total_time/num_epochs:.2f} seconds")
        
        return train_losses, train_metrics_history

    def evaluate(self, test_loader):
        criterion = nn.MSELoss()
        self.model.eval()
        test_loss = 0
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # Move to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Reshape input
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], self.input_size)
                
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Calculate loss
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
                
                # Store predictions
                all_y_true.append(y_batch.cpu().numpy())
                all_y_pred.append(y_pred.cpu().numpy())
        
        # Convert to arrays
        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        
        # Convert normalized values back to original scale
        y_true_original = self.scaler.inverse_transform(all_y_true)
        y_pred_original = self.scaler.inverse_transform(all_y_pred)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true_original, y_pred_original)
        avg_test_loss = test_loss / len(test_loader)
        
        print(f'Test Loss: {avg_test_loss:.6f}, MPA: {metrics["mpa"]:.4f}, '
              f'Direction Accuracy: {metrics["direction_accuracy"]:.4f}')
        
        return avg_test_loss, y_pred_original, y_true_original, metrics

    def predict_future(self, last_sequence, periods=1, period_type='day'):
        if self.last_date is None:
            raise ValueError("Date information not found. Run preprocess_data first.")
        
        # Initialize input sequence
        input_sequence = last_sequence.reshape(-1, 1).copy()
        
        # Days per period mapping
        days_per_period = {
            'day': 1,
            'week': 7,
            'month': 30,
            'quarter': 90,
            'year': 365
        }
        
        if period_type not in days_per_period:
            raise ValueError(f"Invalid period type. Choose from: {list(days_per_period.keys())}")
        
        days_to_add = days_per_period[period_type]
        
        # Predict for each period
        results = []
        current_sequence = input_sequence.copy()
        current_date = self.last_date
        
        for i in range(periods):
            # Scale the current sequence
            scaled_sequence = self.scaler.transform(current_sequence[-self.seq_length:])
            
            # Convert to tensor
            X = torch.FloatTensor(scaled_sequence).reshape(1, self.seq_length, self.input_size).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X)
            
            # Convert back to original scale
            predicted_price = self.scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            
            # Update date
            current_date = current_date + timedelta(days=days_to_add)
            
            # Save result
            results.append({
                'period': i + 1,
                'date': current_date,
                'price': predicted_price[0][0]
            })
            
            # Update sequence for next prediction
            current_sequence = np.vstack([current_sequence[1:], predicted_price])
        
        return {
            'last_actual_date': self.last_date,
            'predictions': results
        }

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'seq_length': self.seq_length,
            'last_date': self.last_date,
            'date_column': self.date_column
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device,weights_only=False)
        
        # Extract information
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.output_size = checkpoint['output_size']
        self.seq_length = checkpoint['seq_length']
        self.scaler = checkpoint['scaler']
        self.last_date = checkpoint.get('last_date', None)
        self.date_column = checkpoint.get('date_column', None)
        
        # Recreate model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        ).to(self.device)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def plot_predictions(self, actuals, predictions, metrics=None, title="Stock Price Prediction", save_path=None):
        plt.figure(figsize=(12, 8))
        
        # Plot actual vs predicted
        plt.subplot(2, 1, 1)
        plt.plot(actuals, label='Actual Price', color='blue')
        plt.plot(predictions, label='Predicted Price', color='red')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # If we have metrics, add a metrics box
        if metrics:
            metrics_text = (f"MSE: {metrics['mse']:.2f}\n"
                           f"RMSE: {metrics['rmse']:.2f}\n"
                           f"MPA: {metrics['mpa']:.4f}\n"
                           f"Dir. Accuracy: {metrics['direction_accuracy']:.4f}")
            
            # Add text box with metrics
            plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot price difference (predicted - actual)
        plt.subplot(2, 1, 2)
        diff = predictions.flatten() - actuals.flatten()
        plt.plot(diff, color='green')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Prediction Error')
        plt.xlabel('Time')
        plt.ylabel('Difference')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Plot saved to {save_path}")
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Error displaying plot: {e}")
                default_path = "prediction_plot.png"
                plt.savefig(default_path)
                plt.close()
                print(f"Plot saved to {default_path}")

    def plot_future_predictions(self, future_results, ticker_symbol=None, save_path=None):
        last_date = future_results['last_actual_date']
        predictions = future_results['predictions']
        
        dates = [pred['date'] for pred in predictions]
        prices = [pred['price'] for pred in predictions]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, marker='o', linestyle='-', color='blue')
        
        title = "Future Stock Price Prediction"
        if ticker_symbol:
            title += f" - {ticker_symbol}"
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Predicted Price")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Plot saved to {save_path}")
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Error displaying plot: {e}")
                default_path = f"future_prediction_{ticker_symbol if ticker_symbol else 'plot'}.png"
                plt.savefig(default_path)
                plt.close()
                print(f"Plot saved to {default_path}")


# Example usage with metrics
def example_usage():
    # Read data
    data = pd.read_csv('VN100_stock_price_1D.csv')
    
    # Filter data for a specific ticker
    ticker_symbol = 'TCB'
    df = data[data['ticker'] == ticker_symbol].copy()
    
    # Use the correct date column name
    date_column = 'data_date'
    
    # Ensure date column exists
    if date_column not in df.columns:
        # Look for alternative date columns if data_date isn't found
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            print(f"Warning: '{date_column}' not found, using '{date_columns[0]}' instead")
            df[date_column] = df[date_columns[0]]
        else:
            print(f"Warning: No date column found, creating synthetic dates")
            df[date_column] = pd.date_range(end=datetime.now().date(), periods=len(df))
    
    # Print some data stats
    print(f"Dataset info for {ticker_symbol}:")
    print(f"- Total records: {len(df)}")
    print(f"- Date range: {df[date_column].min()} to {df[date_column].max()}")
    print(f"- Price range: {df['close_price'].min():.2f} to {df['close_price'].max():.2f}")
    
    # Check for GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        # Set memory allocation
        try:
            # Try to limit GPU memory growth (for TensorFlow compatibility)
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Set TensorFlow to use dynamic GPU memory allocation")
        except:
            pass
    else:
        print("\nNo GPU detected, training will use CPU only")
    
    # Initialize predictor with optimized parameters for GPU
    if gpu_available:
        # Increase batch size and hidden size when using GPU
        predictor = StockPricePredictor(
            seq_length=60,
            hidden_size=128,  # Larger for GPU
            num_layers=2,
            learning_rate=0.001,
            batch_size=128    # Larger for GPU
        )
    else:
        predictor = StockPricePredictor(
            seq_length=60,
            hidden_size=50,   # Smaller for CPU
            num_layers=2,
            learning_rate=0.001,
            batch_size=64     # Smaller for CPU
        )
    
    # Preprocess data
    train_loader, test_loader, train_data, test_data = predictor.preprocess_data(
        df, target_col='close_price', date_col=date_column
    )
    
    print(f"\nLast date in dataset: {predictor.last_date.strftime('%d/%m/%Y')}")
    print(f"Training samples: {len(train_data) - predictor.seq_length}")
    print(f"Testing samples: {len(test_data) - predictor.seq_length}")
    
    # Train model with GPU metrics tracking
    print("\n--- Starting model training ---")
    train_losses, train_metrics = predictor.train(
        train_loader, 
        num_epochs=50,
        use_gpu_metrics=gpu_available
    )
    
    # Evaluate model with metrics
    print("\n--- Evaluating model ---")
    test_loss, predictions, actuals, test_metrics = predictor.evaluate(test_loader)
    
    # Print summary of results
    print("\n--- Training Results ---")
    print(f"Final MPA: {train_metrics[-1]['mpa']:.4f}")
    print(f"Final Direction Accuracy: {train_metrics[-1]['direction_accuracy']:.4f}")
    
    print("\n--- Test Results ---")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MPA: {test_metrics['mpa']:.4f}")
    print(f"Direction Accuracy: {test_metrics['direction_accuracy']:.4f}")
    
    # Plot predictions with metrics
    predictor.plot_predictions(
        actuals,
        predictions,
        metrics=test_metrics,
        title=f"{ticker_symbol} Stock Price Prediction",
        save_path=f"prediction_comparison_{ticker_symbol}.png"
    )
    
    # Get last sequence for future predictions
    last_sequence = df['close_price'].values[-predictor.seq_length:]
    
    # Predict for various time periods
    print("\n--- Future Predictions ---")
    periods = ['day', 'week', 'month', 'quarter']
    for period in periods:
        result = predictor.predict_future(last_sequence, periods=1, period_type=period)
        pred_date = result['predictions'][0]['date']
        pred_price = result['predictions'][0]['price']
        print(f"Prediction for next {period} ({pred_date.strftime('%d/%m/%Y')}): {pred_price:.2f}")
    
    # Predict for next 10 days
    future_10days = predictor.predict_future(last_sequence, periods=10, period_type='day')
    print("\nPredictions for next 10 days:")
    for pred in future_10days['predictions']:
        print(f"Date {pred['date'].strftime('%d/%m/%Y')}: {pred['price']:.2f}")
    
    # Plot future predictions
    predictor.plot_future_predictions(
        future_10days,
        ticker_symbol,
        save_path=f"future_predictions_{ticker_symbol}.png"
    )
    
    # Save and reload model
    print("\n--- Saving and reloading model ---")
    model_path = 'lstm_stock_model.pth'
    predictor.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    new_predictor = StockPricePredictor()
    new_predictor.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    loaded_prediction = new_predictor.predict_future(last_sequence, periods=1, period_type='day')
    loaded_date = loaded_prediction['predictions'][0]['date']
    loaded_price = loaded_prediction['predictions'][0]['price']
    print(f"Prediction from loaded model for {loaded_date.strftime('%d/%m/%Y')}: {loaded_price:.2f}")


if __name__ == "__main__":
    example_usage()