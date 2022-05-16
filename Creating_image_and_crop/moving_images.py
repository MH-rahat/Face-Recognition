import os
import shutil

#os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
image_path_base_in='dataset3/training_set/riyadh/'
image_path_base_out='dataset3/test_set/riyadh/'
for i in range(95,120):
        image_path_in=str(image_path_base_in+str(i)+'.jpg')
        image_path_out=str(image_path_base_out+str(i)+'.jpg')
        shutil.move(image_path_in, image_path_out)
#os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")