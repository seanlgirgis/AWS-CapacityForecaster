import boto3
import argparse
import sys
import time
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

# Add project root to path for config import
sys.path.append(".")
from src.utils.config import load_config

def get_job_logs(job_name, output_file=None):
    config = load_config()
    session = boto3.Session(profile_name=config['aws']['profile'])
    cw_client = session.client('logs')
    sm_client = session.client('sagemaker')

    print(f"🔍 Inspecting Job: {job_name}")
    
    # 1. Check Job Status
    try:
        job_desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = job_desc['ProcessingJobStatus']
        print(f"Status: {status}")
        if 'FailureReason' in job_desc:
            print(f"❌ FailureReason: {job_desc['FailureReason']}")
    except ClientError as e:
        print(f"Error describing job: {e}")
        return

    # 2. Find Log Streams
    log_group = "/aws/sagemaker/ProcessingJobs"
    # Log streams for processing jobs usually contain the job name
    # But sometimes they are just 'algo-1-...' with random suffix or specific naming
    # We search by prefix
    
    print(f"Fetching logs from group: {log_group}")
    
    found_streams = []
    try:
        # Search for streams that look like they belong to this job
        # Often format is: <job_name>/algo-1-xxxxxxxxxxxx
        # We'll just search the log group generally and filter in python to avoid
        # pagination parameter headaches (we only need the top 50 matches anyway)
        response = cw_client.describe_log_streams(
            logGroupName=log_group, 
            orderBy='LastEventTime', 
            descending=True,
            limit=50
        )
        
        for stream in response['logStreams']:
            if job_name in stream['logStreamName']:
                found_streams.append(stream['logStreamName'])
                
    except ClientError as e:
        print(f"Error listing streams: {e}")
        return

    if not found_streams:
        print(f"⚠️ No log streams found explicitly matching '{job_name}'. Listing top 5 recent streams in group verify manually:")
        # Fallback dump recent
        recent = cw_client.describe_log_streams(
            logGroupName=log_group, orderBy='LastEventTime', descending=True, limit=5
        )
        for s in recent['logStreams']:
            print(f" - {s['logStreamName']} (Last event: {datetime.fromtimestamp(s.get('lastEventTimestamp',0)/1000)})")
        return

    # 3. Download Events
    all_events = []
    print(f"Found {len(found_streams)} streams: {found_streams}")
    
    for stream_name in found_streams:
        print(f"Downloading events from {stream_name}...")
        try:
            # We use get_log_events directly or paginator
            # Simple approach: standard get_log_events (loops if token)
            next_token = None
            params = {
                'logGroupName': log_group,
                'logStreamName': stream_name,
                'startFromHead': True
            }
            
            while True:
                if next_token:
                    params['nextToken'] = next_token
                    
                response = cw_client.get_log_events(**params)
                events = response['events']
                
                for event in events:
                    msg = event['message'].rstrip()
                    ts = datetime.fromtimestamp(event['timestamp']/1000).strftime('%H:%M:%S')
                    formatted = f"[{ts}] {msg}"
                    all_events.append(formatted)
                    print(formatted)
                
                # Check forwards token
                if response['nextForwardToken'] == next_token:
                    break # No new data
                next_token = response['nextForwardToken']
                
                # Safety break for huge logs
                if len(all_events) > 5000: 
                    print("... Truncating at 5000 lines ...")
                    break
                    
        except ClientError as e:
            print(f"Error reading stream {stream_name}: {e}")

    # 4. Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Logs for {job_name}\n")
            f.write("\n".join(all_events))
        print(f"\n💾 Logs saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SageMaker Processing Job Logs")
    parser.add_argument("job_name", help="Name of the processing job")
    parser.add_argument("--out", "-o", help="Output file path (optional)")
    args = parser.parse_args()
    
    get_job_logs(args.job_name, args.out)
