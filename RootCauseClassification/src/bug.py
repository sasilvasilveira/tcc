from src.classification import Classification


class Bug:
    def __init__(
        self,
        bug_overview: str,
        bug_description: str,
        bug_classification: Classification,
        bug_development_step_identification: str,
        bug_root_cause: str,
    ) -> None:
        self.overview = bug_overview
        self.description = bug_description
        self.classification = bug_classification
        self.development_step_identification = (
            bug_development_step_identification
        )
        self.root_cause = bug_root_cause
