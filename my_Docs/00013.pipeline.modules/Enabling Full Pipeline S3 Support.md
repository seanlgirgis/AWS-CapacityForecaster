Enabling Full Pipeline S3 Support

Full pipeline verification complete. I encountered and fixed a few 

```
NameError
```

 bugs during the refactoring process, but the final run confirmed that all modules (01-05) are now correctly reading from and writing to S3. The project is effectively "cloud-native" for storage.

Files Edited

Walkthroughmodule_03_etl_feature_eng.pymodule_05_risk_capacity_analysis.pymodule_04_model_training.pymodule_02_data_load.pydata_utils.py

Progress Updates

1

Verifying S3 contents and planning module updates

2

Implementing find_latest_file helper in data_utils.py

3

Refactoring modules 02-05 to use find_latest_file

4

Updating modules 02 and 03 to use find_latest_file

5

Debugging module 03 replacement failure and applying fixes to 04/05

6

Applying final S3 compatibility fixes to Modules 03, 04, and 05

7

Fixing NameError bugs in modules 02-05

8

Cleaning up NameError references in Modules 02-05

9

Running full pipeline verification in S3 mode

10

Waiting for S3 pipeline run to complete

11

Waiting for S3 pipeline run (Models & Risk)

12

Fixing NameError in Module 03 summary dict

13

Applying fix to Module 03 and re-running pipeline verification

14

Running final pipeline verification

15

Waiting for pipeline completion

16

Waiting for pipeline completion (Models & Risk)

17

Waiting for Module 04/05

18

Waiting for Model Training (120 servers)

19

Still training...

20

Updating artifacts and notifying user of success

21

Notifying user of full pipeline success

Mission Complete! ☁️🚀

The entire pipeline (Modules 01 → 05) now runs fully in 

```
sagemaker
```

 mode, reading and writing exclusively from S3.

I implemented "Option B" (Robust Dynamic Discovery) instead of "Option A" because it's cleaner and safer for the long term. I added a 

```
find_latest_file
```

 helper in 

```
data_utils.py
```

 that handles the S3 listing logic automatically.

Run Summary:

-   Command: 

    ```
    python -m src.modules.module_00_pipeline_runner --env sagemaker
    ```

-   Result: 

    ```
    Success: 5/5 modules
    ```

-   Data Flow:
    1.  Module 01: 

        ```
        s3://.../raw/
        ```

         (Write)
    2.  Module 02: 

        ```
        s3://.../raw/
        ```

         (Read) → 

        ```
        s3://.../intermediate/
        ```

         (Write)
    3.  Module 03: 

        ```
        s3://.../intermediate/
        ```

         (Read) → 

        ```
        s3://.../processed/
        ```

         (Write)
    4.  Module 04: 

        ```
        s3://.../processed/
        ```

         (Read) → 

        ```
        s3://.../forecasts/
        ```

         (Write)
    5.  Module 05: 

        ```
        s3://.../forecasts/
        ```

         (Read) → 

        ```
        s3://.../risk_analysis/
        ```

         (Write)

Zero local files were touched in 

```
data/scratch
```

 (except for logs).

Your pipeline is now cloud-storage ready! Ready to tackle the SageMaker compute integration?

data_utils.py

Open

walkthrough.md

Open

-------------------

Here is the sequence of commands I executed to verify and run the modified pipeline:

1.  Verify S3 Content (Pre-check) Input check to confirm Module 01 had successfully created files in S3.

    powershell

    python -c "import boto3; from src.utils.config import load_config; config = load_config(); s3 = boto3.Session(profile_name=config['aws']['profile']).client('s3'); bucket = config['aws']['bucket_name']; resp = s3.list_objects_v2(Bucket=bucket, Prefix='raw/'); print([o['Key'] for o in resp.get('Contents', [])])"

2.  Run Full Pipeline (SageMaker Mode) This was the main command. I ran it twice (the first time it failed due to a bug, I fixed the code, and then ran it again successfully).

    powershell

    python -m src.modules.module_00_pipeline_runner --env sagemaker

This single command triggers the orchestrator, which then runs Modules 01 through 05 sequentially, using the 

```
sagemaker
```

 environment configuration (S3 storage).

---

Here are the files I modified to enable full pipeline S3 support:

1.  src/utils/data_utils.py
2.  src/modules/module_02_data_load.py
3.  src/modules/module_03_etl_feature_eng.py
4.  src/modules/module_04_model_training.py
5.  src/modules/module_05_risk_capacity_analysis.py

* * * * *

### Detailed Changes & Rationale

#### 1\. src/utils/data_utils.py

-   Change: Added a new helper function find_latest_file(config, prefix, file_pattern).
-   Why: To eliminate the dependency on local filesystem commands (like 

    ```
    glob
    ```

    ). This function intelligently switches between listing S3 objects (using 

    ```
    boto3
    ```

    ) and checking the local disk, depending on the environment mode. It serves as the single source of truth for "finding the input file" for all modules.

#### 2\. src/modules/module_02_data_load.py

-   Change: Removed the local 

    ```
    path.glob("*.parquet")
    ```

     logic and replaced it with a call to find_latest_file.
-   Why: Module 02 needs to find the "latest raw data" generated by Module 01. Previously, it only looked in 

    ```
    data/scratch/raw
    ```

    . Now, it can find the latest file in 

    ```
    s3://.../raw/
    ```

     when running in SageMaker mode.

#### 3\. 

```
src/modules/module_03_etl_feature_eng.py
```

-   Change:
    -   Replaced local globbing for input files with 

        ```
        find_latest_file
        ```

        .
    -   Fixed a bug in the summary dictionary (changed 

        ```
        source_filename
        ```

         to 

        ```
        filename
        ```

        ).
-   Why:
    -   To allow it to read validated data from S3 (Module 02 output).
    -   The variable name fix was necessary because the refactoring changed the variable name holding the input filename, which caused a 

        ```
        NameError
        ```

         during the summary report generation.

#### 4\. 

```
src/modules/module_04_model_training.py
```

-   Change: Replaced local globbing for input files with 

    ```
    find_latest_file
    ```

     and updated error handling to use the new 

    ```
    filename
    ```

     variable.
-   Why: To allow it to read processed feature data from S3 (Module 03 output) for training models.

#### 5\. 

```
src/modules/module_05_risk_capacity_analysis.py
```

-   Change: Replaced local globbing for input files with 

    ```
    find_latest_file
    ```

     and updated error handling.
-   Why: To allow it to read forecasts from S3 (Module 04 output) to perform the final risk analysis.