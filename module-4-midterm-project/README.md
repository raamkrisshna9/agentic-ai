docker run -d \
  --name genai-redis \
  -p 6379:6379 \
  redis/redis-stack-server:latest

conda create --prefix ./my_env python=3.13 -y
conda activate ./my_env
pip install -r requirements.txt --upgrade