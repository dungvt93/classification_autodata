# First, you should install flickrapi
# pip install flickrapi

import flickrapi
import urllib.request
from PIL import Image

# Flickr api access key
flickr=flickrapi.FlickrAPI('e4d932ab10f24980ac4f2f553557a562', 'c4f3c55e601eb81e', cache=True)


keyword = 'siberian husky'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=100,           # may be you can try different numbers..
                     sort='relevance')

urls = []
for i, photo in enumerate(photos):
    print (i)

    url = photo.get('url_c')
    urls.append(url)

    # get 50 urls
    if i > 50:
        break

print (urls)

# Download image from the url and save it to '00001.jpg'
urllib.request.urlretrieve(urls[1], '00001.jpg')

# Resize the image and overwrite it
image = Image.open('00001.jpg')
image = image.resize((256, 256), Image.ANTIALIAS)
image.save('00001.jpg')