conda create --prefix ./my_env python=3.11 -y
conda activate ./my_env
pip install -r requirements.txt --upgrade

https://colab.research.google.com/drive/1p1v4H4f5w3Tj9BVxr7uUtJwB1TA-QxFR?usp=sharing


docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
​
