
import boto3
import sys

def get_logs():
    session = boto3.Session(profile_name='study', region_name='us-east-1')
    client = session.client('logs')
    
    log_group = '/aws/sagemaker/ProcessingJobs'
    job_name = 'proc-job-1769883852'
    
    # Dynamically find the stream
    streams = client.describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=job_name,
        limit=1
    )
    
    if not streams['logStreams']:
        print("No log streams found yet.")
        return

    log_stream = streams['logStreams'][0]['logStreamName']
    
    print(f"Fetching logs from {log_stream}...")
    
    response = client.get_log_events(
        logGroupName=log_group,
        logStreamName=log_stream,
        limit=500,
        startFromHead=False
    )
    
    events = response['events']
    print(f"Retrieved {len(events)} events.")
    
    with open('full_logs.txt', 'w', encoding='utf-8') as f:
        for event in events:
            msg = event['message'].rstrip()
            print(msg)
            f.write(msg + '\n')

if __name__ == "__main__":
    get_logs()
