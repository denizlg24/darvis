from darvis import daemon
from darvis.daemon import run

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f'[ERROR] There was a problem starting DARVIS: {e}')
        traceback.print_exc()

