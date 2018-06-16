from app import *

@app.route('/simple_nn/index')
def simple_nn_index():
    return render_template('simple_nn/index.html')
