# VidStab-Compare-Sample
[VidStab](https://github.com/AdamSpannbauer/python_video_stab) の各キーポイント抽出方法を比較し、動画を出力するサンプルです。

https://github.com/Kazuhito00/VidStab-Compare-Sample/assets/37477845/425f9597-f548-4c7f-8e1f-705d8ad683d9

# Requirement 
* opencv-contrib-python 4.9.0.80 or later
* vidstab 1.7.4 or later

# Usage
```
python main.py
```
* --movie<br>
動画ファイルの指定<br>
デフォルト：指定なし　※指定しない場合はサンプル動画をダウンロード
* --output<br>
出力動画のファイル名<br>
デフォルト：output.mp4
* --smoothing_window<br>
平滑窓の長さ<br>
デフォルト：30

# License 
VidStab-Compare-Sample is under [MIT license](LICENSE).

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
