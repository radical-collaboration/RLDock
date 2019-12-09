

config['env'] = 'lactamase_docking'
agent = get_agent_class('PPO')(env='lactamase_docking', config=config)
agent.restore(checkpoint)