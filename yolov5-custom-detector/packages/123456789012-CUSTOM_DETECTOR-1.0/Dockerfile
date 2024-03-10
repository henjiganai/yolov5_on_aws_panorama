FROM public.ecr.aws/panorama/panorama-application
WORKDIR /panorama
COPY . .
RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev libffi-dev python-dev python3-dev gcc python3.7-dev
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
