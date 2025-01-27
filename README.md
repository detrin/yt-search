# yt-search
Search YouTube videos and their context.


```
# list subs
yt-dlp --list-subs https://www.youtube.com/watch\?v\=QlbuY24alSk
# download EN subs
yt-dlp --write-subs --sub-lang en --convert-subs srt --skip-download https://www.youtube.com/watch?v=QlbuY24alSk
# doanload autogenerated EN subs
yt-dlp --write-auto-subs --sub-lang en --convert-subs srt --skip-download https://www.youtube.com/watch?v=QlbuY24alSk
```