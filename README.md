# KnowledgeDistillation

This repository is a composite of multiple repos used during the project. All jobs scripts are written for LSF 10 HPC schedular but are relatively simple to port to systems like Slurm

All jobs and configs assume that you are using a single H100 and a 32 core cpu with 5 gigs of memory per core 

## Minillm

All files related to knowledge distillation are in the minillm folder, except for the scipts for downloading the data. To see examples of how to use these files, see the folders bloom_jobs and llama_jobs where the download jobs for each respective dataset specifies how to execute the download scripts. The *_jobs folders also contain all of the jobs for running the distillationss

##  LMeval and TK-instruct

In either folder you will find jobs scripts in job folders that detail the specific experiments outlined in the report. The readme's in either folder detail how to get the specifik task data for the evaluations as well. 