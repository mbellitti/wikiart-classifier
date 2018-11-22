"""this script definies a flow_from_dataframe_remote method, that can be
activated in any other script by executing this file.
"""
import os
import os.path
from keras import preprocessing
from keras_preprocessing.image import *

def load_img_remote(path, url, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest', save_img = False):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        save_img: If path doesn't exist, i.e. the file does not exist locally
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    ###change method to fetch image!
    #img_raw is the output of response.content of the image url
    img_raw = ############
    img = pil_image.open(BytesIO(img_raw))
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def flow_from_dataframe_remote(self, dataframe, directory,
                        x_col="filename", y_col="class", has_ext=True,
                        target_size=(256, 256), color_mode='rgb',
                        classes=None, class_mode='categorical',
                        batch_size=32, shuffle=True, seed=None,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='png',
                        subset=None,
                        interpolation='nearest',
                        sort=True,
                        drop_duplicates=True):
    """Takes the dataframe and the path to a directory
     and generates batches of augmented/normalized data.
    # A simple tutorial can be found at: http://bit.ly/keras_flow_from_dataframe
    # Arguments
        dataframe: Pandas dataframe containing the filenames of the
            images in a column and classes in another or column/s
            that can be fed as raw target data.
        directory: string, path to the target directory that contains all
            the images mapped in the dataframe.
        x_col: string, column in the dataframe that contains
            the filenames of the target images.
        y_col: string or list of strings,columns in
            the dataframe that will be the target data.
        target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
            The dimensions to which all images found will be resized.
        color_mode: one of "grayscale", "rgb". Default: "rgb".
            Whether the images will be converted to have 1 or 3 color channels.
        classes: optional list of classes (e.g. `['dogs', 'cats']`).
            Default: None. If not provided, the list of classes will be
            automatically inferred from the `y_col`,
            which will map to the label indices, will be alphanumeric).
            The dictionary containing the mapping from class names to class
            indices can be obtained via the attribute `class_indices`.
        class_mode: one of "categorical", "binary", "sparse",
            "input", "other" or None. Default: "categorical".
            Determines the type of label arrays that are returned:
            - `"categorical"` will be 2D one-hot encoded labels,
            - `"binary"` will be 1D binary labels,
            - `"sparse"` will be 1D integer labels,
            - `"input"` will be images identical
                to input images (mainly used to work with autoencoders).
            - `"other"` will be numpy array of `y_col` data
            - None, no labels are returned (the generator will only
                yield batches of image data, which is useful to use
            `model.predict_generator()`, `model.evaluate_generator()`, etc.).
        batch_size: size of the batches of data (default: 32).
        shuffle: whether to shuffle the data (default: True)
        seed: optional random seed for shuffling and transformations.
        save_to_dir: None or str (default: None).
            This allows you to optionally specify a directory
            to which to save the augmented pictures being generated
            (useful for visualizing what you are doing).
        save_prefix: str. Prefix to use for filenames of saved pictures
            (only relevant if `save_to_dir` is set).
        save_format: one of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        follow_links: whether to follow symlinks inside class subdirectories
            (default: False).
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
            If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
            supported. If PIL version 3.4.0 or newer is installed, `"box"` and
            `"hamming"` are also supported. By default, `"nearest"` is used.
        sort: Boolean, whether to sort dataframe by filename (before shuffle).
        drop_duplicates: Boolean, whether to drop duplicate rows
            based on filename.
    # Returns
        A DataFrameIterator yielding tuples of `(x, y)`
        where `x` is a numpy array containing a batch
        of images with shape `(batch_size, *target_size, channels)`
        and `y` is a numpy array of corresponding labels.
    """

    return DataFrameIterator_remote(dataframe, directory, self,
                             x_col=x_col, y_col=y_col,
                             #has_ext=has_ext, #removed for remote use
                             target_size=target_size, color_mode=color_mode,
                             classes=classes, class_mode=class_mode,
                             data_format=self.data_format,
                             batch_size=batch_size, shuffle=shuffle, seed=seed,
                             save_to_dir=save_to_dir,
                             save_prefix=save_prefix,
                             save_format=save_format,
                             subset=subset,
                             interpolation=interpolation,
                             sort=sort,
                             drop_duplicates=drop_duplicates)

class DataFrameIterator_remote(Iterator):
    """Iterator capable of reading images from a directory on disk
        through a dataframe. If the image is not on disk it is fetched from
        remote disk through url given.
    # Arguments
        dataframe: Pandas dataframe containing the filenames of the
                   images in a column and classes in another or column/s
                   that can be fed as raw target data.
        directory: In this use with remote, is the directory
            under which all local images are present and where downloaded
            may be saved.
            Data in x_col must not be full path.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        x_col: Column in dataframe that contains all the filenames (or absolute
            paths, if directory is set to None).
        y_col: Column/s in dataframe that has the target data.
        z_col: Column in dataframe that has the url of the file
        has_ext: bool, Whether the filenames in x_col has extensions or not.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of
            each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `"other"`: targets are the data(numpy array) of y_col data
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        sort: Boolean, whether to sort dataframe by filename (before shuffle).
        drop_duplicates: Boolean, whether to drop duplicate rows based on filename.
    """self.fail('message')
    def __init__(self, dataframe, directory, image_data_generator,
                 x_col="filenames", y_col="class", z_col = "url",
                 #has_ext=True, #removed for remote use
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 sort=True,
                 drop_duplicates=True):
        super(DataFrameIterator, self).common_init(image_data_generator,
                                                   target_size,
                                                   color_mode,
                                                   data_format,
                                                   save_to_dir,
                                                   save_prefix,
                                                   save_format,
                                                   subset,
                                                   interpolation)
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Install pandas to use flow_from_dataframe.')
        if type(x_col) != str:
            raise ValueError("x_col must be a string.")
        if type(has_ext) != bool:
            raise ValueError("has_ext must be either True if filenames in"
                             " x_col has extensions,else False.")
        self.df = dataframe.copy()
        if drop_duplicates:
            self.df.drop_duplicates(x_col, inplace=True)
        self.x_col = x_col
        self.df[x_col] = self.df[x_col].astype(str)
        self.directory = directory
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', 'other', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             '"other" or None.')
        self.class_mode = class_mode
        self.dtype = dtype
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            if class_mode not in ["other", "input", None]:
                classes = list(self.df[y_col].unique())
        else:
            if class_mode in ["other", "input", None]:
                raise ValueError('classes cannot be set if class_mode'
                                 ' is either "other" or "input" or None.')
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # Second, build an index of the images.
        self.ids = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        self.urls = list(self.df[z_col])

        if self.directory is not None:
            if not os.path.isdir(self.directory):
                os.mkdir(self.directory)
                #TODO: make warning
        else:
            raise ValueError('Directory must be specified.')
        self.ids = list(self.df[x_col])


        if self.split:
            num_files = len(self.ids)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            self.df = self.df.iloc[start: stop, :]
            self.ids = self.ids[start: stop]

        if class_mode not in ["other", "input", None]:
            classes = self.df[y_col].values
            self.classes = np.array([self.class_indices[cls] for cls in classes])
        elif class_mode == "other":
            self.data = self.df[y_col].values
            if type(y_col) == str:
                y_col = [y_col]
            if "object" in list(self.df[y_col].dtypes):
                raise TypeError("y_col column/s must be numeric datatypes.")
        self.samples = len(self.ids)
        if self.num_classes > 0:
            print('Found %d images belonging to %d classes.' %
                  (self.samples, self.num_classes))
        else:
            print('Found %d images.' % self.samples)

        super(DataFrameIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.ids[j] + '.jpg'
            url = self.urls[j]
            if self.directory is not None:
                img_path = os.path.join(self.directory, fname)
            else:
                img_path = fname
            #here we have to include the routine to fetch the picture#########
            img = load_img_remote(img_path, url,
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'other':
            batch_y = self.data[index_array]
        else:
            return batch_x
        return batch_x, batch_y

    def _list_valid_filepaths(self, white_list_formats):

        def get_ext(filename):
            return os.path.splitext(filename)[1][1:].lower()

        df_paths = self.df[self.x_col]

        format_check = df_paths.map(get_ext).isin(white_list_formats)
        existence_check = df_paths.map(os.path.isfile)

        valid_filepaths = list(df_paths[np.logical_and(format_check,
                                                       existence_check)])

        return valid_filepaths

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

if __name__ = '__main__':
    preprocessing.ImageDataGenerator.flow_from_dataframe_remote =flow_from_dataframe_remote
