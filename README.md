segmentation analysis for different segment lengths
````
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=5 --name repeated-segment-2-5 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_5.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=10 --name repeated-segment-2-10 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_10.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=15 --name repeated-segment-2-15 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_15.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=20 --name repeated-segment-2-20 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_20.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=25 --name repeated-segment-2-25 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_25.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=30 --name repeated-segment-2-30 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_30.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=35 --name repeated-segment-2-35 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_35.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=40 --name repeated-segment-2-40 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_40.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=3 -e training_iters=2000000 -e first_segment_length=45 --name repeated-segment-2-45 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segmentation-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/repated_segment_2_45.txt 2>&1
````
docker build -t segment-analysis .

````
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=10 -e training_iters=2000000 -e first_segment_length=17 -e second_segment_length=17 --name three-segments -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segment-count-analysis  >> /Users/mehmetpekmezci/ozu/segment-analysis/segment_count.txt 2>&1
````

````
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=10 -e training_iters=2000000 --name segment_analysis_2 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segment-analysis python two-segment-train.py  >> /Users/mehmetpekmezci/ozu/segment-analysis/segment_count_2.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=10 -e training_iters=2000000 --name segment_analysis_3 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segment-analysis python three-segment-train.py  >> /Users/mehmetpekmezci/ozu/segment-analysis/segment_count_3.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=10 -e training_iters=2000000 --name segment_analysis_4 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segment-analysis python four-segment-train.py  >> /Users/mehmetpekmezci/ozu/segment-analysis/segment_count_4.txt 2>&1
docker run -d  -e PYTHONUNBUFFERED=1 -e repeat_count=10 -e training_iters=2000000 --name segment_analysis_5 -v /Users/mehmetpekmezci/ozu/segment-analysis:/app/output segment-analysis python five-segment-train.py  >> /Users/mehmetpekmezci/ozu/segment-analysis/segment_count_5.txt 2>&1
````

