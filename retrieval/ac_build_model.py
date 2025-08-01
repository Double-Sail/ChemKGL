from serch_model import INSTANCE as GRAPH_ENTITY_SEARCHER
import threading

if __name__ == '__main__':
    model = GRAPH_ENTITY_SEARCHER

    threading.Thread(
        target=model.build,
        # kwargs={"description": description}
    ).start()
