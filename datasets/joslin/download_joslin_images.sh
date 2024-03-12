#!/bin/bash
input_file="./img_names_subset.txt"
num_imgs=$(cat "$input_file" | wc -l)

cnt=0
while IFS= read -r line;
do 
    cnt=$((cnt+1));
    /media/enc/vera1/sebastian/codes/Foundational-Multimodal-Fusion-Benchmark/google-cloud-sdk/bin/gcloud storage cp gs://arvo_2022_images/$line "./Joslin_images"; 
    echo "$cnt of $num_imgs files processed; running ..."
done < "$input_file"
echo "$cnt of $num_imgs files processed; finished :-)"