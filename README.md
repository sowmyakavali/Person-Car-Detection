# Person-Car-Detection

### Here I have used YOLOV5 model to detect and classify objects.

    > Go through this notebook to understand person and car detection from a image.
 https://colab.research.google.com/drive/1OF34YigkLjrGq92vSKaxAFLywfZ0mXpx?usp=sharing
    
## Different Segments of project.

    1. Data Collection
    2. Data Processing
    3. Training
    4. Validation
    5. Inference
    6. Results
    
 ### 1. Data Collection
 
        You can download dataset from here (https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz)
        
        License: These images are taken from OpenImages. All annotations are licensed under CC BY  4.0 and the images have a CC BY 2.0 license*
        
        Dataset Details:   
        
            Number of images: 2239 
            Size: 750 MB (approx.) 
            Categories: 2 (person and car) 
            Annotation format: COCO 
            
        Data directory structure: 
         data 
         |_ annotations 
         |_ bbox-annotations.json 
         |_ images 
         |_ image_000000001.jpg 
         |_ image_000000002.jpg 
         |_ ... 
         ... 
         
 ### 2. Data Preprocessing
        > Here the annotations are in coco format , so i need to convert them into yolov5 pytorch format. you can do it by  using the below function.
        
              convert_labels(coco, images_path, output_path)
            
        > After convertion the annotations will be in .txt format and each file will having the normalized values of x-center, y-center, width, height.
        
        > Split the dataset into train, test & val sets.
        
        > You can plot the histogram of labels count using this function.
        
              plot_labels_hist(labels, labels_names)             
              
        > Becauze of the data is so imabalanced , i am doing augmentation to make it balanced or near by equal count. Remember do augmentation for only training samples.
        
            augment_for_specific_classes(classes_dict, train_images_path, train_labels_path, countofinstances, augmethodsids)
            
            countofinstances : How many instances you wanted to increase
            augmethodsids : there are 15 augmentation methods were written so you can give list of integers ranging from 0 to 14
            
        > Create a data.yaml file by declaring the all the paths and labels.
        
              d = {
                       'train': train_images,
                       'val': val_images,
                       'test':test_images,
                       'nc': len(labels_names),
                       'names': labels_names
                   }  # dictionary

              with open(config_file, 'w') as f:
                      yaml.dump(d, f)
            
### 3. Training

        > first clone the yolov5 repo and install all requirements.
        
              !git clone https://github.com/ultralytics/yolov5  
              %cd yolov5
              %pip install -qr requirements.txt
              
        > start training.
        
              !python3 train.py --img 640 --batch 32 --epochs 500 --data /content/DATASET/data.yaml --weights yolov5s.pt --patience 25 --save-period 10 --cache
              
        > Here I have used yolov5s.pt pretrained weights and did transfer learning.
        
### 4. Validation

        > Do model validation on validation set.
        
              !python val.py --weights /content/drive/MyDrive/Assignment/weights/best.pt --data /content/DATASET/data.yaml --img 640 --iou 0.65 --batch-size 32 --project /content/drive/MyDrive/Assignment/val --verbose --half
              
        > Here give same batch size and image size whatever you given for training.
         
### 5. Testing (Inference)

        > Check your model performance on unseen data.
        
              !python detect.py --weights /content/drive/MyDrive/Assignment/weights/best.pt --img 640 --conf 0.5 --source /content/DATASET/test/images --project /content/drive/MyDrive/Assignment/detect
              
### 6. Results 

          > Here you can check my training summary.
          
              https://drive.google.com/drive/folders/1-VlQzlX-CUq0ULXKPqMUWpQVCrUUUG1C?usp=sharing
              
          > We can further imrpove the model by 
          
                  1. increasing the dataset.
                  2. Using Balanced dataset.
                  3. Training model for more epochs.
        
        
