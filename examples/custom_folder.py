from datasets.image_test_data import ImageDataSetLoader

loader = ImageDataSetLoader(
    'C:/Users/felix/OneDrive/octimine/Testing/English/Fnt',
    grayscale=True,
    image_size=[28, 28],
    one_hot=True)

loader.convert_to_tf_record()
    
