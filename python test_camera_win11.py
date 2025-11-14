import cv2

print("Testing camera...")

# Try multiple indexes automatically
for idx in range(5):
    print(f"\nTrying camera index {idx}...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"SUCCESS: Camera opened at index {idx}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break
            cv2.imshow(f"CAMERA {idx}", frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        print(f"FAILED: Camera not available at index {idx}")

print("\nTest finished.")
