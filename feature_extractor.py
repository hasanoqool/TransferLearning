import glob
import os
import pathlib
import h5py
import numpy as np
import sklearn.utils as skutils
from sklearn.preprocessing import LabelEncoder
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm


class FeatureExtractor(object):
    def __init__(self,model,input_size,label_encoder, num_instances,
    feature_size, output_path, features_key='features', buffer_size=1000):
    
        
        if os.path.exists(output_path):
                error_msg = (f'{output_path} already exists. Please delete it and try again.')
                raise FileExistsError(error_msg)
    
        self.model = model
        self.input_size = input_size
        self.le = label_encoder
        self.feature_size = feature_size
        self.buffer_size = buffer_size

        #self.buffer will contain a buffer of both instances and labels, 
        # while self.current_index will point to the next free location 
        # within the datasets in the inner HDF5 database #        
        self.buffer = {'features': [], 'labels': []}
        
        self.current_index = 0

        self.db = h5py.File(output_path, 'w')
        self.features = self.db.create_dataset(features_key,(num_instances,feature_size),dtype='float')
        self.labels = self.db.create_dataset('labels',(num_instances,),dtype='int')


    def extract_features(self,image_paths,labels,batch_size=64,shuffle=True):
            if shuffle:
                image_paths, labels = skutils.shuffle(image_paths,labels)
            encoded_labels = self.le.fit_transform(labels)
            self._store_class_labels(self.le.classes_)

            for i in tqdm(range(0, len(image_paths),batch_size)):
                batch_paths = image_paths[i: i + batch_size]
                batch_labels = encoded_labels[i:i + batch_size]
                batch_images = []

                for image_path in batch_paths:
                    image = load_img(image_path,
                    target_size=self.input_size)
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    image = imagenet_utils.preprocess_input(image)
                    batch_images.append(image)

                batch_images = np.vstack(batch_images)
                feats = self.model.predict(batch_images, batch_size=batch_size)
                new_shape = (feats.shape[0], self.feature_size)
                feats = feats.reshape(new_shape)
                self._add(feats, batch_labels)
            
            self._close()
        

    def _add(self, rows, labels):
            self.buffer['features'].extend(rows)
            self.buffer['labels'].extend(labels)
            if len(self.buffer['features']) >= self.buffer_size:
                self._flush()



    def _flush(self):
            next_index = (self.current_index + len(self.buffer['features']))
            buffer_slice = slice(self.current_index, next_index)
            self.features[buffer_slice] = self.buffer['features']
            self.labels[buffer_slice] = self.buffer['labels']
            self.current_index = next_index
            self.buffer = {'features': [], 'labels': []}


    def _store_class_labels(self, class_labels):
            data_type = h5py.special_dtype(vlen=str)
            shape = (len(class_labels),)
            label_ds = self.db.create_dataset('label_names', shape, dtype=data_type)
            label_ds[:] = class_labels
        

    def _close(self):
            if len(self.buffer['features']) > 0:
                self._flush()
            self.db.close()


files_pattern = pathlib.Path("/content/drive/MyDrive/Colab Notebooks/Computer Vision Repos/datasets/car_ims/*.jpg")
files_pattern = str(files_pattern)
input_paths = [*glob.glob(files_pattern)]


output_path = pathlib.Path("/content/drive/MyDrive/Colab Notebooks/Computer Vision Repos/datasets/car_ims/car_ims_rotated")
if not os.path.exists(str(output_path)):
    os.mkdir(str(output_path))


labels = []
output_paths = []
for index in tqdm(range(len(input_paths))):
    image_path = input_paths[index]
    image = load_img(image_path)
    rotation_angle = np.random.choice([0, 90, 180, 270])
    rotated_image = image.rotate(rotation_angle)
    rotated_image_path = str(output_path /
    f'{index}.jpg')
    rotated_image.save(rotated_image_path, 'JPEG')
    output_paths.append(rotated_image_path)
    labels.append(rotation_angle)
    image.close()
    rotated_image.close()


features_path = str(output_path / 'features.hdf5')
model = VGG16(weights='imagenet', include_top=False)
fe = FeatureExtractor(model=model,input_size=(224, 224, 3),label_encoder=LabelEncoder(),
num_instances=len(input_paths),feature_size=512 * 7 * 7,output_path=features_path)


fe.extract_features(image_paths=output_paths,labels=labels)