# yt-search
Search YouTube videos and their context.


The following will serach term `kubernetes` and download subtitles in srt format. However not all videos have en subtitles, we will first download autogenerated subtitles and then regular subtitles to override the autogenerated subtitles if they exist.
```
search_phrase="kubernetes"
yt-dlp "ytsearch10:${search_phrase}" --write-auto-subs --sub-lang en --convert-subs srt --skip-download
yt-dlp "ytsearch10:${search_phrase}" --write-subs --sub-lang en --convert-subs srt --skip-download
```

Make sure you have installed `ollama` on your system, then pull `llama3.1:8b` and make sure that it is running
```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama run llama3.1:8b "What is the capital of France?"
```

The following will process the YT subtitles and answer the defined question.
```
question="What is BERT?"
# download subtitles from YT
bash download_yt_subtitles.sh -s "$question" -n 10 -o data/tmp_srt
# convert subtitles to txt
bash convert_srt_to_txt.sh data/tmp_srt data/tmp_txt
# run the RAG and find the answer
python rag.py \
    --input_folder data/tmp_txt \
    --question "$question" \
    --top_k "$TOP_K" \
    --max_recursion "$MAX_RECURSION"
# cleaning
rm -r data/tmp_*
```

## Docker
You can also run the whole app using `docker-compose`
```
docker-compose up --build 
```
This will open gradio app on http://localhost:3000
![](https://i.imgur.com/rMCLXT9.png)
The question can take around 4 mins to process depending on your hardware.

```mermaid
%% Docker Compose Service Architecture
graph TD
    Redis["Redis
    Image: redis:alpine
    Ports: 6379:6379
    Healthcheck: redis-cli ping every 5s"]
    
    Volume_ollama([Volume: ollama])
    
    Ollama["Ollama
    Image: ollama/ollama
    Ports: 11434:11434
    Runs llama3.1:8b model"]
    
    Backend["Backend
    Build: ./backend
    Ports: 8000:8000
    Env: API_USER, API_PASS, OLLAMA_HOST"]
    
    Gradio["Gradio App
    Build: ./gradio_app
    Ports: 3000:7860"]
    
    Backend -->|"depends_on (service_healthy)"| Redis
    Gradio -->|depends_on| Backend
    Ollama -->|mounts| Volume_ollama
    Backend -->|"OLLAMA_HOST=ollama:11434"| Ollama
    Backend --> Redis

    classDef service stroke:#333,stroke-width:2px;
    classDef volume stroke:#333,stroke-width:2px;
    class Redis,Ollama,Backend,Gradio service;
    class Volume_ollama volume;
```
