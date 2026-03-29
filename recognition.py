import torch
import cv2
import torchvision.transforms as transforms
from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

model = EmotionCNN(num_classes=len(classes))
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)

        emotion = classes[predicted.item()]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()