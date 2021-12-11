from project_questions import ProjectQuestions
import os


if __name__ == "__main__":
    vo_data = {}
    vo_data['dir'] = r"..\..\..\Q2_data"
    vo_data['sequence'] = 5
    project = ProjectQuestions(vo_data)
    project.run()