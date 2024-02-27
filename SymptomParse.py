#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 00:05:41 2024

@author: jkrek
"""

import asyncio
import aiohttp
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

# Improved token caching integrated with aiohttp.ClientSession
class TokenCache:
    def __init__(self):
        self.token = None
        self.expires = None

    async def get_token(self, session):
        if self.token and self.expires > asyncio.get_event_loop().time():
            return self.token

        client_id = '7f1ee356-4634-4ef6-9193-3554dc4801c5_7ca1da3e-227d-444e-a7ef-76b1713d8743'
        client_secret = 'FAbOtLbFMvAtQi/GkhlFvA4vsXrBGMECc5PNEuYbBhM='
        token_url = 'https://icdaccessmanagement.who.int/connect/token'
        data = {
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'client_credentials'
        }

        async with session.post(token_url, data=data) as response:
            if response.status == 200:
                response_data = await response.json()
                self.token = response_data.get('access_token')
                self.expires = asyncio.get_event_loop().time() + response_data.get('expires_in', 3600) - 300  # Refresh 5 min early
                return self.token
            else:
                raise Exception(f"Failed to obtain token: {response.status}")

# Function to fetch diseases for a given symptom and chapter
async def fetch_diseases_for_symptom(session, token_cache, symptom, chapter, semaphore, regex_pattern):
    try:
        access_token = await token_cache.get_token(session)
    except Exception as e:
        print(f"Error obtaining token: {e}")
        return symptom, []

    headers = {
        'Authorization': f'Bearer {access_token}',
        'API-Version': 'v2',
        'Accept': 'application/ld+json',
        'Accept-Language': 'en'
    }
    api_url_template = 'https://id.who.int/icd/release/11/2023-01/mms/search'
    chapter_filter = f"&chapterFilter={chapter:02d}"
    api_url = f"{api_url_template}?q={symptom}&useFlexisearch=true&medicalCodingMode=false&propertiesToBeSearched=Definition{chapter_filter}"

    async with semaphore, session.get(api_url, headers=headers) as api_response:
        if api_response.status == 200:
            response_data = await api_response.json()
            disease_names = [regex_pattern.sub('', entity['title']) for entity in response_data.get('destinationEntities', [])]
            return symptom, disease_names
        else:
            print(f"Error calling API for symptom '{symptom}', chapter {chapter}: {api_response.status}")
            return symptom, []

# Main asynchronous function
async def main():
    symptoms = ['cough', 'pain']  # Example symptoms
    chapters = range(1, 3)  # Example chapter range
    token_cache = TokenCache()
    semaphore = asyncio.Semaphore(10)  # Limiting to 10 concurrent requests
    regex_pattern = re.compile('<[^<]+?>')  # Compiled regex pattern for HTML tag stripping

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_diseases_for_symptom(session, token_cache, symptom, chapter, semaphore, regex_pattern) for symptom in symptoms for chapter in chapters]
        results = await asyncio.gather(*tasks)

        disease_occurrences = Counter([disease for _, diseases in results for disease in diseases])
        disease_df = pd.DataFrame(disease_occurrences.most_common(), columns=['Disease', 'Occurrences'])

        if not disease_df.empty:
            disease_df.to_csv('/Users/jkrek/Downloads/disease_occurrences_optimized.csv', index=False)
            print("Disease occurrences saved to 'disease_occurrences_optimized.csv'.")

           
            top_n = 20  # Adjustable parameter for top N diseases
            top_diseases = disease_df.head(top_n)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(top_diseases['Disease'], top_diseases['Occurrences'], color='darkred')
            ax.set_xlabel('Occurrences')
            ax.set_title(f'Top {top_n} Most Likely Disease Matches')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = loop.create_task(main())
    else:
        loop.run_until_complete(main())
