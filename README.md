## How to find key geoscience terms in text without mastering NLP using Amazon Comprehend


Geosemantics is the application of linguistic techniques to geoscience. Geoscientists often have access to more reports than they can reasonably read so they are commonly challenged in filtering through reports to find relevant information (for example, this report about the [Wolfcamp and Bone Spring shale plays](https://www.eia.gov/maps/pdf/Wolfcamp_BoneSpring_EIA_Report_July2019.pdf)). Traditional Natural Language Processing (NLP) techniques such as Named Entity Recognition (NER) must be trained to identify geologically relevant terms. 

[Amazon Comprehend](https://aws.amazon.com/comprehend/) provides a suite of NLP tools and pre-trained models for common tasks such as NER. While models are trained for general use, they are designed to be extensible for domain-specific text. 

In this post, we build a custom entity recognizer using Comprehend through the [AWS SDK for Python](https://aws.amazon.com/sdk-for-python/) (Boto3). 

### Stratigraphic named entity recognition

Stratigraphy is a branch of geology concerned with the study of rock layers and layering. Stratigraphic intervals are the result of global and local conditions. These intervals have known rock properties that are used to reduce uncertainty about the subsurface. Exploration geology reports provide information about the stratigraphy, biological markers (biostratigraphy) and associated geological age (chronostratigraphy). 

## Approach to building a NER

Named Entity Recognition is built by supervised machine learning models. In order to train the model, the scientist needs a training text and annotation for the terms of interest, often provided as a key-value entity list. The [British Geological Survey](https://github.com/BritishGeologicalSurvey/geo-ner-model) (BGS) built an NER model using Stanford's CoreNLP system. Their training and testing data is [publicly available](https://github.com/BritishGeologicalSurvey/geo-ner-model). 

We desire to train the NER to classify entities within the text so we can use this information for further analysis. The BGS entity list identifies general geological terms (labeled LEXICON), chorostratigraphic terms, and biostrategraphic terms (labeled BIOZONE). For this analysis, we focused only on chronostratigraphic terms. 

The training document contains lines from a geological report. For example:

```
All glacigenic deposits mapped in NO76NW are assigned to the Mearns
Glacigenic Subgroup of the Caledonia Glacigenic Group.
This subgroup, which is equivalent to the Mearns Drift Group of Merritt
et al., comprises Mill of Forest Till, Ury Silts, and Drumlithie Sand
And Gravel formations.
The highly irregular middle zone is underlain by Permian and Triassic
rocks; the Permian strata include Upper Permian Zechstein sedimentary
rocks that locally crop out in the study area.
```

The entity list file is a structured list of entities including:

```
Text, Type
Permian, CHRONOSTRAT
Triassic, CHRONOSTRAT
Lenisulcata, CHRONOSTRAT
```

The data must follow this specific format. You need three documents: Training text, testing text, and entity list. We will use the BSG's text and entity labels formatted for use with Comprehend. 

## Walkthrough

The code and support files are available [here](https://github.com/orgs/aws-samples/teams/global-sa-energy-nlp). You can also use this AWS CloudFormation template to create all the resources needed for this project in your account. Alternatively, you can download and unzip the dataset onto your computer from the [amazon-comprehend-geoscience-nlp.zip](https://github.com/aws-samples/amazon-comprehend-energy-geoscience-nlp/blob/master/amazon-comprehend-geoscience-nlp.zip) file.

In this example, we create a custom entity recognizer to extract information about geologic eras. To train a custom entity recognition model, choose one of two ways to provide data to Amazon Comprehend: [Annotations](https://docs.aws.amazon.com/comprehend/latest/dg/cer-annotation.html) or [Entity lists](https://docs.aws.amazon.com/comprehend/latest/dg/cer-entity-list.html). In this example, we use the entity list method. I removed variations of the name such as "Age", "Epoch", and "Eon."  BGS provides data segmented into training and test sets. These files plus the entity list must be uploaded to Amazon S3 and Comprehend must have permission to access the S3 bucket through a service-linked IAM role. 

Using a Python Jupyter Notebook in Amazon SageMaker, execute the code below to create an Amazon Comprehend custom entity training job.  This snippet assumes that the training, test, and entity documents are in the same SageMaker folder as the Jupyter Notebook. 

```python
import boto3
import uuid

comprehend = boto3.client("comprehend")
role =
"arn:aws:iam::141317253884:role/service-role/AmazonComprehendServiceRole-comprehend"

bucket = "personalizelab-chicago"
entity_types = "CHRONOSTRAT"
train_documents = "bgs-geo-training-data.txt"
test_documents = "bgs-geo-testing-data.txt"
entity_list = "bgs-geo-entity-list.txt"
files = [train_documents, test_documents, entity_list]

s3 = boto3.resource('s3')
[s3.Bucket(bucket).upload_file(file, str(file)) for file in files]

response = comprehend.create_entity_recognizer(
    RecognizerName="geo-entity-{}".format(str(uuid.uuid4())),
    LanguageCode="en",
    DataAccessRoleArn= role,
    InputDataConfig={
        "EntityTypes": [
            {
                "Type": entity_types
            }
        ],
        "Documents": {
            "S3Uri": '/'.join(['s3:/', bucket,
train_documents])
        },
        "EntityList": {
            "S3Uri": '/'.join(['s3:/', bucket, entity_list])
        }
    }
)
recognizer_arn = response["EntityRecognizerArn"]
```

This cell executes three key tasks.

1.  It copies the training, test, and entity documents to Amazon S3. 
2.  It designs an Amazon Comprehend custom entity extraction training job and directs Amazon Comprehend to access the training and entity documents from Amazon S3.
3.  It beings the training job that takes 20-25 minutes to train the model. 

You can check the status of the training job every 60 seconds using this
code. Once the status is "TRAINED" you can proceed to the next step.

```python
import time

while True:
    response = comprehend.describe_entity_recognizer(
        EntityRecognizerArn=recognizer_arn
    )

    status = response["EntityRecognizerProperties"]["Status"]
    if "IN_ERROR" == status:
        sys.exit(1)
    if "TRAINED" == status:
        break

    time.sleep(60)
```
Optional: 

Amazon Simple Notification Service can send you a text message once the
training is complete with this code.

```python
# Optional code to send a text message once the training is complete

# number for the scientist. Must include the international code ("+1" for the US)
phone_number = "+14055551234" 

# Create an SNS client
sns = boto3.client("sns")

sns.publish(
    PhoneNumber = phone_number,
    Message = "{} training has stopped with status
{}".format(response["RecognizerName"], status)
)
```

Once the training is complete, Amazon Comprehend reports accuracy metrics for the entities in the training dataset.  The classification metrics indicate the model correctly recognized the custom entity in 99.17% of instances. The proportion of actual positives that are correctly identified by the model was 98.36%. 

| Metric     |    Value  | Definition |   
|----------- |---------- |------------|
|Precision   |99.17      |Positive predictive value|
|Recall      |98.36      |True positive rate|
|F-1 Score   |98.76      |Harmonic mean of the precision and recall|

# Test your model

To test the model, we create a detection job. We provide a few parameters including the format of the test document and where to save the results. When the detection job is complete, Comprehend will save the results as JSON files in your output S3 bucket path.

```python
response = comprehend.start_entities_detection_job(
    EntityRecognizerArn=recognizer_arn,
    JobName="Detection-Job-Name-{}".format(str(uuid.uuid4())),
    LanguageCode="en",
    DataAccessRoleArn=role,
    InputDataConfig={
        "InputFormat": "ONE_DOC_PER_LINE",
        "S3Uri": '/'.join(['s3:/', bucket, test_documents])
    },
    OutputDataConfig={
        "S3Uri": '/'.join(['s3:/', bucket, "output"])
    }
)
```

As before, you can choose to receive an Amazon SNS message once the detection job is complete.

```python
# Optional code to send a text message once the detection job is
complete

while True:
    response = comprehend.describe_entities_detection_job(
        EntityRecognizerArn=recognizer_arn
    )

    status =
response["EntitiesDetectionJobProperties"]["Status"]
    if "IN_ERROR" == status:
        sys.exit(1)
    if "COMPLETED" == status:
        break

    time.sleep(60)
    
sns.publish(
    PhoneNumber = phone_number,
    Message = "{} job has stopped with status
    {}".format(response["JobName"],
    response["EntitiesDetectionJobProperties"]["JobStatus"])
)
```

You might now notice that Amazon Comprehend has picked up additional
words with varying spellings. This is how Comprehend differs from a
simple text look up. Comprehend is using a probabilistic model based on
natural language processing to identify chronostratigraphic terms. 

Example input:

The inliers afford some insight into the development of early
Phanerozoic basins in northern Britain but the geological setting of the
inliers, in terms of the relative positions of lithospheric plates in
Silurian times, remains conjectural.

Example Response: 

```python
{"Entities": [{"BeginOffset": 74, "EndOffset": 85, "Score":
0.9982181191444397, "Text": "Phanerozoic", "Type":
"CHRONOSTRAT"}, {"BeginOffset": 241, "EndOffset": 249, "Score":
0.9996685981750488, "Text": "Silurian", "Type":
"CHRONOSTRAT"}], "File": "NER-test-data.txt", "Line": 5}
```

The results show:

-   Offset: The beginning and end of the offset is the number of characters into the line when the word appears.
-   Score: Comprehend's confidence that the identified text is of the specified type ranging from 0-1.
-   Type: The type of the entity extracted based on the training model. 

## Extension

Amazon Comprehend can be used for batch inference, as we have done here, or for real-time inference as described in this [blog post](https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-amazon-comprehend-custom-entity-recognition-real-time-endpoints/). 

## Conclusion

In this post, we built a custom entity recognition model to identify geologic periods without using NLP frameworks. The Amazon Comprehend response provides metadata that can be used for filtering geoscience documentation. Combined with a search index like Amazon ElastiCache or Apache Solr, these results could substantially reduce the time geoscientists spend search for data in reports. 

Future projects could extend this to additional entity types or apply text extraction methods to convert PDF reports into tabular data with the appropriate metadata about the geological age. This model can scale to analyze documents of arbitrary length. 

The workflow this type of analysis is typically batch, but the model could be extended to provide near-realtime inference by deploying the model as an Endpoint. 

Try custom entities now from the [Amazon Comprehend console](https://console.aws.amazon.com/comprehend/home) and get detailed instructions in the [Amazon Comprehend documentation](https://docs.aws.amazon.com/comprehend/latest/dg/auto-ml.html). This solution is available in all Regions where Amazon Comprehend is available. Please refer to the [AWS Region Table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) for more information.



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

The data is licensed under CC BY-SA 4.0. See the THIRD PARTY LICENSE file. 

