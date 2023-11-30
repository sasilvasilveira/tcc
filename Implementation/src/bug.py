class Bug:
    def __init__(
        self,
        bug_overview: str,
        bug_description: str,
        bug_category: str,
        bug_development_step_identification: str
    ) -> None:
        self.overview = bug_overview
        self.description = bug_description
        self.category = bug_category
        self.development_step_identification = bug_development_step_identification
        self.root_cause = ""
