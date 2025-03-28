import torch

print(f"PyTorch phiên bản: {torch.__version__}")
print(f"CUDA khả dụng: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Số lượng GPU: {torch.cuda.device_count()}")
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"Phiên bản CUDA trong PyTorch: {torch.version.cuda}")
    
    # Kiểm tra hiệu suất GPU với phép nhân ma trận đơn giản
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    
    # Đo thời gian
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c = torch.matmul(a, b)
    end.record()
    
    torch.cuda.synchronize()
    print(f"Thời gian thực hiện phép nhân ma trận: {start.elapsed_time(end)} ms")
else:
    print("Không tìm thấy GPU hỗ trợ CUDA.")