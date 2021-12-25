from project_questions import ProjectQuestions


def main():
    """
    main entry point to start the visual odometry algorithm after picking the dataset
    """
    vo_data = {}
    vo_data['dir'] = r"..\..\..\Q2_data"
    vo_data['sequence'] = 5
    project = ProjectQuestions(vo_data)
    project.run()


if __name__ == "__main__":
    main()