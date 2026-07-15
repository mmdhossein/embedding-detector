import json
import yaml
from lxml import html
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def extract_json_value(data, path):
    """Pure Python deep extraction for dot-notation paths."""
    if not path: return []
    keys = path.split('.')

    def traverse(obj, key_queue):
        if not key_queue:
            return obj
        current_key = key_queue[0]

        if isinstance(obj, dict):
            return traverse(obj.get(current_key), key_queue[1:])
        elif isinstance(obj, list):
            # Recurse through arrays and flatten the results
            extracted = [traverse(item, key_queue) for item in obj]
            result = []
            for res in extracted:
                if isinstance(res, list):
                    result.extend(res)
                elif res is not None:
                    result.append(res)
            return result
        return None

    return traverse(data, keys)

def extract_values(payload_item, query, doc_type):
    """Unified extractor for both JSON traversal and XPath."""
    if doc_type == "HTTP_DOC":
        try:
            tree = html.fromstring(payload_item)
            return [str(r).strip() for r in tree.xpath(query) if str(r).strip()]
        except Exception as e:
            logging.error(f"XPath extraction failed: {e}")
            return []

    elif doc_type == "API_METRICS":
        res = extract_json_value(payload_item, query)
        return res if isinstance(res, list) else [res] if res else []

    return []

def get_raw_data(url):
    with open(url, 'r', encoding='utf-8') as f:
        return f.read()

def harvest_metrics(yaml_config_path):
    """Harvests and normalizes metrics into a strict structural format per item."""
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    doc_type = config.get('type')
    raw_content = get_raw_data(config.get('url'))
    
    # Ensure payload is iterable. API_METRICS iterates over 'results', HTTP_DOC iterates once over the raw string.
    if doc_type == "API_METRICS":
        parsed_data = json.loads(raw_content)
        payload_items = parsed_data.get("results", []) if isinstance(parsed_data, dict) else parsed_data
        if not isinstance(payload_items, list):
            payload_items = [payload_items]
    else:
        payload_items = [raw_content]

    aggregated_metrics = []

    for metric in config.get('metrics', []):
        metric_id = metric.get('id', 'unknown_metric')
        extract_rules = metric.get('extract', {})
        
        # Iterate over each item (each dictionary in results) to create separate metric blocks
        for item in payload_items:
            metric_struct = {
                "metric_id": metric_id,
                "channel": "unknown",
                "transaction_type": "unknown",
                "time_list": [],
                "success_list": [],
                "failed_list": []
            }

            for target_key, rule in extract_rules.items():
                query = rule.get('query')
                if query and target_key in metric_struct:
                    metric_struct[target_key] = extract_values(item, query, doc_type)

            # Flatten scalar metadata (channel, transaction_type)
            for scalar in ['channel', 'transaction_type']:
                if metric_struct[scalar] and isinstance(metric_struct[scalar], list):
                    metric_struct[scalar] = metric_struct[scalar][0]

            # Only append if we actually found something for this block, avoiding ghost metrics
            if metric_struct['time_list'] or metric_struct['success_list']:
                aggregated_metrics.append(metric_struct)

    return aggregated_metrics

if __name__ == "__main__":
    yaml_file = './rial.yaml'
    
    try:
        results = harvest_metrics(yaml_file)
        logging.info(f"Total metric blocks accumulated: {len(results)}")
        logging.info(f"Output Payload:\n{json.dumps(results, indent=2)}")
    except Exception as e:
        logging.error(f"Failed to harvest metrics: {e}")
