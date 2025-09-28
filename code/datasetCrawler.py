import json
from types import SimpleNamespace
from pybliometrics.scopus import ScopusSearch
import pandas as pd
import csv
from tkinter import EXCEPTION
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import csv
from scholarly import scholarly
from scholarly import ProxyGenerator


def extractScopus(origin, destination):
    """
    WARNING: you need to set up your API key from Scopus first.
    Receives a CSV file with DOIs in the first column and 
    extracts relevant info from SCOPUS API writing it in destination as
    a CSV file."""
    count = 0
    with open(origin,'r',encoding='utf-8') as csvfile:
        f = open(destination,'a',newline='',encoding='utf-8')
        writer = csv.writer(f)
        csv_reader = csv.reader(csvfile)
        header = []
        header.append('Year')
        writer.writerow(header)
        for row in csv_reader:
            doi = row[0]
            try:
                count += 1
                if doi != '[]' and doi != '-1' and doi != 'N/A':
                    authorCount, citedby_count, aggregationType, doi, abstract = query(doi)
                    row.append(abstract)
                    print(count)
                    writer.writerow(row)
                    f.flush()
                else:
                    row.append('N/A')
                    print(count)
                    writer.writerow(row)
                    f.flush()
            except Exception as e: 
                print(e)
                row.append('N/A')
                writer.writerow(row)
                print(doi,sep='\t')
                print(count)
                f.flush()
        f.flush()
        f.close()
                

def query(title):
    try:
        s = ScopusSearch(title)
        df = pd.DataFrame(pd.DataFrame(s.results))
        return df.at[0,'title'],df.at[0,'description'],df.at[0,'coverDate'][:4],df.at[0,'citedby_count'],df.at[0,'author_count'],df.at[0,'aggregationType'],df.at[0,'coverDate'][0:4]
    except:
        return 'ERROR','ERROR','ERROR','ERROR','ERROR','ERROR'

def selenium(url:str,driver)->int:
    doi="N/A"
    if "ieeexplore.ieee.org/stamp/stamp" in url:
        numerourl= url.split('=')[1]
        driver.get("https://ieeexplore.ieee.org/document/"+numerourl+"/metrics#metrics")
        element="0"
        try:
            try:            
                elements = driver.find_elements(By.CLASS_NAME,"stats-document-abstract-doi")
                for x in elements: 
                    link=x.find_element(By.TAG_NAME,'a')
                    doi = link.text                
            except:
                return doi
        except:
            print("Timeout")
        return doi
    else:
        return doi

def scrapIEEEdoi(origin, destination):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    count = 0
    incorrect = 0
    with open(origin,'r') as csvfile:
        f = open(destination,'w',newline='',encoding='utf-8')
        writer = csv.writer(f)
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        header.append('doi')
        writer.writerow(header)
        
        for row in csv_reader:
            doi=selenium(row[3],driver)
            if doi == "N/A":
                incorrect = incorrect + 1
            count = count + 1
            row.append(doi)
            print(row)
            print(count)
            writer.writerow(row)

        print("Incorrect scrapped: ", end=" ")
        print(incorrect)
        f.flush()
        f.close()
    driver.close()


def extractScholarly():
    dois = ['10.1109/ICCSNT.2013.6967127','10.1109/ISCC-C.2013.45''10.1109/FMCAD.2013.7035522''10.5594/j18356XY','10.1109/ITMC.2013.7352708']
    search_query = scholarly.search_pubs('10.1145/3238147.3240477')
    data = next(search_query)
    print(data['bib']['title'])
    print(data['bib']['abstract'])
    print(data['num_citations'])
    print(len(data['bib']['author']))



def bibtexParser(bibSelected, csvWithoutSelected, destination):
    """Receives a CSV with DOIs (first column) and checks if the DOI is labeled as yes 
    (is selected as candidate) using the bibtext file that contains the selected rows. 
    Result is the original csv but completed with the label."""
    with open(bibSelected, "r", errors="ignore") as f:
        lines = f.readlines()
        dois = list(filter(lambda doi: 'doi={' in doi, lines))
        f = open(destination,'w',newline='',encoding='utf-8')
        writer = csv.writer(f)
        with open(csvWithoutSelected,'r',encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                doi = 'doi={'+row[0]+'}\n'
                if doi in dois:
                    row.append('yes')
                else:
                    row.append('no')
                writer.writerow(row)
            f.flush()
            f.close()
        