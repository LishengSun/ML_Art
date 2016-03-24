# Get images via
# google image search engine


import urllib
import mechanize
from bs4 import BeautifulSoup
import os

def getPic_google(painter, tot_num):
	painter_dir = 'data/'+painter.replace(' ', '_')+'/'
	if not os.path.exists(painter_dir):
		os.makedirs(painter_dir)
	painter_save = painter.replace(' ', '_')
	painter = painter.replace(' ', '%20')

	# try:
	browser = mechanize.Browser()
	browser.set_handle_robots(False)
	browser.addheaders = [('User-agent', 'Mozilla')]

	i=0
	while i<tot_num/20:
		start = 20*i
		htmltext = browser.open('https://www.google.com/search?site=imghp&tbm=isch&source=hp&biw=1280&bih=635&q='+painter+'&oq='+painter+'&start='+str(start))
		soup = BeautifulSoup(htmltext, 'html.parser')
		results = soup.findAll('img')
		j=0
		for r in results:
			img_url = r['src']
			urllib.urlretrieve(img_url, './'+painter_dir+painter_save+str(start+j)+'.jpg')#+painter+'/')#+painter+str(start+j)+'.jpg')
			j+=1
		i += 1
	
# if __name__ == "__main__":
# 	getPic_google('Raphael painting', 40)

