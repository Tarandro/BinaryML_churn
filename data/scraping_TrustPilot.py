from bs4 import BeautifulSoup
from selenium import webdriver
import regex as re
import pandas as pd
import time
import random as rd
from selenium.common.exceptions import WebDriverException

# SCRAPER
def TrustPilot_Scraper(base_url, max_pages, size):
    """ Scraper function : initialize a browser, for each page get html page with BeautifulSoup,
        get each reviews of the page and meta information
    Args:
        base_url (str) : base of the url page
        max_pages ('all' or int)
        size (int): number of reviews by pages
    Return:
        df (dataframe) : columns : 'title', 'date', 'star' and 'text'
    """
    browser = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver')
    load_content = browser.implicitly_wait(30)

    # Define empty lists to be scraped
    comment_title = []
    comment_date = []
    comment_star = []
    comment_text = []
    nbre_loser = 0

    if max_pages == 'all':
        url = base_url + "?page=%d" % (1)
        browser.get(url)
        load_content
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        search_results_count = soup.find('span', {'class': 'headline__review-count'}).text
        # space is translated by '\xa', keep only number:
        number_of_reviews = ''
        for i in search_results_count:
            if i in [str(j) for j in range(10)]:
                number_of_reviews += i
        max_pages = int(number_of_reviews) // size
        time.sleep(rd.choice(range(10, 30)))

    # iterate over pages and extract
    for page in range(1, max_pages + 1):
        print("Page %d" % page)

        url = base_url + "?page=%d" % (page)

        for i in range(5):
            try:
                browser.get(url)
                print('browser get url.')
                break
            except WebDriverException:
                print('Quit and initialize new browser')
                browser.quit()
                browser = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver')
            if i == 5:
                print('error browser')


        load_content
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        search_results = soup.find('div', {'class': 'review-list'})
        contents = search_results.find_all('div', {'class': 'review-card'})

        for content in contents:
            try:
                title = content.find('h2').text
                title = title.replace('  ', '')
                title = title.replace('\n','')
                print(title)

                comment = content.find('p')
                if comment != None: #pas forcément de texte
                    comment = comment.text
                    comment = comment.replace('  ', '')
                    comment = comment.replace('\n', '')

                html = content.find('script', {'data-initial-state': 'review-dates'})
                list_date = [item.group(1) for item in re.finditer('publishedDate":"(.+?)"', str(html))]
                if len(list_date) > 0:
                    date = list_date[0]
                else:
                    date = None

                html = content.find('div', {'class': 'star-rating star-rating--medium'})
                list_star = [int(alt[-1]) for alt in re.findall('alt=".', str(html))]
                if len(list_star) > 0:
                    star = list_star[0]
                else:
                    star = None

                comment_title.append(title)
                comment_date.append(date)
                comment_star.append(star)
                comment_text.append(comment)

            except:
                print("loser")
                nbre_loser += 1
                continue

        print("Number of document fail : " + str(nbre_loser) + "/" + str(size * page))
        if page != max_pages:
            t_sleep = rd.choice(range(3, 6))
            print("Time sleep " + str(t_sleep) + "s for Page " + str(page) + "/" + str(max_pages))
            time.sleep(t_sleep)
        print("-----")

    # Save in DF
    df = pd.DataFrame()
    df['title'] = comment_title
    df['date'] = comment_date
    df['star'] = comment_star
    df['text'] = comment_text

    # print("Complete")
    browser.quit()

    return df


if __name__ == '__main__':
    # Call Function - Scrape and save data
    # Bank :
    # - orangebank
    # - floabank
    # - cofidis
    base_url = 'https://fr.trustpilot.com/review/cofidis.fr'
    TrustPilot_data = TrustPilot_Scraper(base_url=base_url, max_pages='all', size=20)
    TrustPilot_data.to_csv('./data/TrustPilot_cofidis.csv', index=False)