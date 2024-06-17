from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "fc"
    tasks = {
        "algorithms" : ["zhang_mean_fc_nosig"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=True)
    summary, details = ev.evaluate()
