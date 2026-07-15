import yaml
import requests
import logging
from lxml import html
from jsonpath_ng import parse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def extract_values(document, query_config, doc_type):
    q_type = query_config.get('type')
    query = query_config.get('query', '')
    
    if q_type == 'static':
        return [query_config.get('value')]
        
    if doc_type == 'HTTP_DOC' and q_type == 'xpath':
        # lxml xpath execution
        nodes = document.xpath(query)
        return [str(node.text_content() if hasattr(node, 'text_content') else node).strip() for node in nodes]
        
    if doc_type == 'API_METRICS' and q_type == 'jsonpath':
        # jsonpath_ng execution
        jsonpath_expr = parse(query)
        return [match.value for match in jsonpath_expr.find(document)]
        
    return []

def harvest_metrics(yaml_files):
    unified_metrics = []

    for file_path in yaml_files:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        url = config['url']
        doc_type = config['type']
        extract_rules = config['extract']
        
        logging.info(f"Interrogating {url} [{doc_type}]")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Prepare the carcass for dissection
            if doc_type == 'HTTP_DOC':
                document = html.fromstring(response.content)
            else:
                document = response.json()
                
            # Extract raw organs
            extracted_data = {
                key: extract_values(document, rule, doc_type)
                for key, rule in extract_rules.items()
            }
            
            # Normalize list lengths (padding static/short lists)
            max_len = max((len(v) for v in extracted_data.values()), default=0)
            for k, v in extracted_data.items():
                if len(v) == 1 and max_len > 1:
                    extracted_data[k] = v * max_len # Duplicate static values (like channel or time)
                elif len(v) < max_len:
                    extracted_data[k] = v + [0] * (max_len - len(v)) # Null fallback
                    
            # Stitch them together
            for i in range(max_len):
                metric = {key: extracted_data[key][i] for key in extracted_data}
                metric['source'] = url
                unified_metrics.append(metric)
                
            logging.info(f"Successfully harvested {max_len} metrics from {url}.")
            
        except Exception as e:
            logging.error(f"Failed to harvest from {url}: {str(e)}")

    return unified_metrics

# Execute the harvest
# configs = ['switch1.yaml', 'api.yaml', 'switch2.yaml', 'json.yaml']
# results = harvest_metrics(configs)
# logging.info(f"Total metrics accumulated: {len(results)}")
