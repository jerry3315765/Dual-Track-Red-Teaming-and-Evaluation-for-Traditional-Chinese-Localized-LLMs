
class Generator:
    def __init__(self):
        pass
        
def generate_prompts(prompts_config):
    """
    Combines intents and templates to create full prompts.
    """
    prompts = []
    intents = prompts_config.get('intents', [])
    templates = prompts_config.get('templates', {})
    
    for intent in intents:
        for key, tmpl_info in templates.items():
            template_str = tmpl_info['template']
            # Simple string replacement for now
            full_prompt = template_str.replace('{question}', intent)
            prompts.append({
                "template_id": key,
                "intent": intent,
                "prompt": full_prompt
            })
    return prompts
