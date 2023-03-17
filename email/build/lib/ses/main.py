import boto3
from botocore.exceptions import ClientError


SENDER = "Rahul Yedida Notifications <noreply@ryedida.me>"
RECIPIENT = "rahul@ryedida.me"

AWS_REGION = "us-east-1"

# The character encoding for the email.
CHARSET = "UTF-8"

# Create a new SES resource and specify a region.
client = boto3.client('ses',region_name=AWS_REGION)


def send_email(subject, message):
    body_text = message
    body_html = f"<html><body><p>{message}</p></body></html>"

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': body_html,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': body_text,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': subject,
                },
            },
            Source=SENDER
        )
    # Display an error if something goes wrong.    
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
