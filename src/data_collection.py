import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from urllib.parse import urljoin
import os

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourseScraperAV:
    def __init__(self):
        self.base_url = "https://courses.analyticsvidhya.com"
        self.courses_url = urljoin(self.base_url, "/collections/courses")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_course_links(self, page_url):
        try:
            response = self.session.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all course cards using the correct class
            course_cards = soup.find_all('a', class_='course-card')
            
            logger.info(f"Found {len(course_cards)} course cards")
            
            course_links = []
            for card in course_cards:
                # Extracting course information
                href = card.get('href')
                if href:
                    course_url = urljoin(self.base_url, href)
                    title = card.find('h3').get_text(strip=True) if card.find('h3') else None
                    lesson_count = card.find('span', class_='course-card__lesson-count')
                    lesson_count = lesson_count.get_text(strip=True) if lesson_count else "Unknown"
                    
                    # Get image URL
                    img_elem = card.find('img', class_='course-card__img')
                    image_url = img_elem.get('src') if img_elem else None
                    
                    course_links.append({
                        'url': course_url,
                        'title': title,
                        'lesson_count': lesson_count,
                        'image_url': image_url
                    })
                    logger.debug(f"Found course: {title}")

            return course_links

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {page_url}: {str(e)}")
            return []

    def parse_course_page(self, course_info):
        try:
            response = self.session.get(course_info['url'])
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Debug information
            logger.debug(f"Parsing course page: {course_info['url']}")

            # Try to find description
            description = None
            desc_elements = soup.select('.course-description, .description, .course-summary, .course__description')
            if desc_elements:
                description = desc_elements[0].get_text(strip=True)

            # Try to find curriculum
            curriculum = []
            curr_elements = soup.select('.course-curriculum, .curriculum, .course-content, .curriculum__section-title')
            for elem in curr_elements:
                curriculum.append(elem.get_text(strip=True))

            return {
                'title': course_info['title'],
                'url': course_info['url'],
                'lesson_count': course_info['lesson_count'],
                'image_url': course_info['image_url'],
                'description': description or "No description available",
                'curriculum': ' | '.join(curriculum) if curriculum else "No curriculum available"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error parsing course page {course_info['url']}: {str(e)}")
            return None

    def scrape_courses(self, max_pages=9):  
        all_courses = []
        page = 1

        while page <= max_pages:
            page_url = f"{self.courses_url}?page={page}"
            logger.info(f"Scraping page {page}: {page_url}")
            
            course_links = self.get_course_links(page_url)
            
            if not course_links:
                logger.info(f"No courses found on page {page}. Ending scrape.")
                break

            for course_info in course_links:
                course_data = self.parse_course_page(course_info)
                if course_data:
                    all_courses.append(course_data)
                time.sleep(2)  # Polite delay between requests

            page += 1
            time.sleep(1)  # Delay between pages

        return all_courses

def save_courses_to_csv(courses, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df = pd.DataFrame(courses)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(courses)} courses to {filepath}")

def main():
    scraper = CourseScraperAV()
    courses = scraper.scrape_courses()
    
    if not courses:
        logger.warning("No courses were found. Please check the website structure or connection.")
        return
    
    # Save to CSV and JSON 
    csv_path = '../data/courses_data.csv'
    json_path = '../data/courses_data.json'
    
    save_courses_to_csv(courses, csv_path)
    
    # save as JSON 
    df = pd.DataFrame(courses)
    df.to_json(json_path, orient='records', indent=2)
    logger.info(f"Saved {len(courses)} courses to {json_path}")

if __name__ == "__main__":
    main()