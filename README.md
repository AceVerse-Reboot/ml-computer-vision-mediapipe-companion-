# MediaPipe Companion

This browser project uses Google's MediaPipe Tasks Vision package.

Features:
- Live webcam
- MediaPipe Gesture Recognizer
- Hand landmark drawing
- Gesture confidence
- Stabilised 0–5 finger estimate
- MediaPipe Face Landmarker
- Visible facial-movement labels using blendshape scores

Important:
The expression labels describe visible facial movements only. They do not determine a person's true internal emotion.

## Run

1. Open this folder in Visual Studio Code.
2. Start Live Server.
3. Open `index.html`.
4. Click **Start Camera**.
5. Allow camera access.

You need an internet connection on first load because the MediaPipe JavaScript package and model files are loaded from Google's/CDN servers.
