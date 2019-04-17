FROM dev_basic:1
COPY requirements.txt /home/
RUN pip install -r /home/requirements.txt

# Substitute this line with your own model repository or with a copy from local:
ADD http://localhost/model_v1/model.pkl /home/model.pkl 

COPY *.py /home/
WORKDIR /home/
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000","wsgi:app"]
CMD ["--workers=2"]

#ENV FLASK_APP=service
#ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]
