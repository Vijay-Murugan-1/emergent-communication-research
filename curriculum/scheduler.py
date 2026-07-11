class CurriculumScheduler:
    """
    Progressively increases the difficulty of the communication task.
    Stage 1: Warmup (500 episodes) - No communication penalty.
    Stage 2: Continuous (5000) - Continuous communication.
    Stage 3: Discrete (50000) - Discrete communication.
    Stage 4: Noise (200000) - Noise introduced.
    Stage 5: Full Curriculum.
    """
    def __init__(self, config):
        self.config = config
        self.current_stage = 1
        
    def step(self, episode: int):
        if episode < 500:
            self.current_stage = 1
        elif episode < 5000:
            self.current_stage = 2
        elif episode < 50000:
            self.current_stage = 3
        elif episode < 200000:
            self.current_stage = 4
        else:
            self.current_stage = 5
            
    def get_stage_config(self):
        # Depending on the stage, we can yield dynamic configurations
        # overriding the base config properties.
        return {
            "stage": self.current_stage,
            "penalties_enabled": self.current_stage >= 2,
            "force_continuous": self.current_stage == 2,
            "noise_prob": self.config.communication.noise_prob if self.current_stage >= 4 else 0.0
        }
