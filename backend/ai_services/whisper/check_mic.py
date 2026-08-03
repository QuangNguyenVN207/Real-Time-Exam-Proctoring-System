import sounddevice as sd

print("=== DANH SÁCH THIẾT BỊ ÂM THANH ===")
print(sd.query_devices())

try:
    default_input = sd.query_devices(kind='input')
    print("\n" + "="*50)
    print(f"🎤 THIẾT BỊ MẶC ĐỊNH ĐANG ĐƯỢC CHỌN: {default_input['name']}")
    print(f"🎛️ Tần số Hz: {default_input['default_samplerate']}")
    print(f"🔌 Số kênh (Channels): {default_input['max_input_channels']}")
    print("="*50)
except Exception as e:
    print(f"\n[LỖI] Không thể tìm thấy Micro mặc định: {e}")