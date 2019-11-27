
class ImageLoader:
  
  	def __init__(self):
  		import os
  		from tqdm import tqdm_notebook
  		import numpy as np
  		from PIL import Image

		self.images_all = None

  	def read_all(folder_path, key_prefix=""):
	    '''
	    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
	    '''
	    print("Reading:")
	    images = {}
	    files = os.listdir(folder_path)
	    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):
	        file_path = os.path.join(folder_path, file_name)
	        image_index = key_prefix + file_name[:-4]
	        image = Image.open(file_path)
	        image = image.convert("L")
	        images[image_index] = np.array(image.copy()).flatten()
	        image.close()
	    return images

	def load_images(keyword_list=[]):
		self.keyword_list = keyword_list
		for keyword in self.keyword_list:
		    self.images_all = read_all("../input/level_1_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))
		print("Loaded ",len(self.images_all)," images.\n")
		return images_all

