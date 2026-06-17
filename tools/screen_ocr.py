import pyautogui
import winocr


def capture_and_ocr() -> str:
    try:
        screenshot = pyautogui.screenshot()
        result = winocr.recognize_pil_sync(screenshot)
        text = result.get("text", "").strip()
        if not text:
            return "[No text detected on screen]"
        return text
    except Exception as e:
        return f"[OCR error: {e}]"
