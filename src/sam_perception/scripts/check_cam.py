import cv2

def list_cameras(max_check=10):
    print("🔍 正在扫描可用摄像头...")
    available_cams = []
    
    for index in range(max_check):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"✅ 发现设备 /dev/video{index}: 分辨率 {w}x{h}")
                available_cams.append(index)
            cap.release()
        else:
            pass
            
    print("-" * 30)
    if not available_cams:
        print("❌ 未找到任何摄像头，请检查 USB 连接！")
    else:
        print(f"📸 建议尝试的设备 ID: {available_cams}")
        print("通常笔记本自带的是 0，新插的 USB 摄像头可能是 2 或 4。")

if __name__ == "__main__":
    list_cameras()