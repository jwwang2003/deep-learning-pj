# External training data

## Object detection data

External training data should be downloaded a placed here.

Unzip the aoi.zip only and place the contents in this folder. Ignore the aoi-train and aoi-val compressed zip files.

Included a large collection of background/null images \(unlabelled images\). These images
are not supposed to have any relevant detection labels, this can help the model:
- reduce false positives
- learn the required features better \(maybe faster\)

## Segmentation data

Note that the actual images are stored on Amazon S3. Please run the `s3_data` script to fetch the images to local.

- Latest dataset version: `collection2`
    - 884 labelled masks
- Version 1 data: `collection1`
    - Around 300 labelled masks
- Version 2 data: `collection2` \(+ `collection1`\)
    - Around 450 labelled masks 
