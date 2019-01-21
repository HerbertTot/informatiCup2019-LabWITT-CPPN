    hideSpinner();

    $('#myModal').modal({show: false});
    var socket = new WebSocket("ws://localhost:8008/");

    startSocket();


    function uuidv4() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            let r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function hideSpinner() {
        let x = document.getElementById("spinner");
        x.style.display = "none";
    }

    function showSpinner() {
        let x = document.getElementById("spinner");
        x.style.display = "block";
    }

    function appendImages(src) {
        let img = document.createElement("img");
        img.src = src;
        img.classList.add("img-thumbnail");

        let div = document.getElementById('images');
        div.appendChild(img);
    }

    function fillDropdown(options) {
        let select = document.getElementById("target_class");
        for(let i = 0; i < options.length; i++) {
            let opt = options[i];
            let el = document.createElement("option");
            el.textContent = opt;
            el.value = opt;
            select.appendChild(el);
        }

        document.getElementById('target_class').disabled = false;
        document.getElementById('target_conf').disabled = false;
        document.getElementById('grayscale').disabled = false;
        document.getElementById('rgb').disabled = false;
        document.getElementById('init').disabled = false;
        document.getElementById('max_queries').disabled = false;
        document.getElementById('start_button').disabled = false;

    }

    function disableForm(fileinput) {
        if(fileinput) {
            document.getElementById('image_file').disabled = true;
        }
        document.getElementById('target_class').disabled = true;
        document.getElementById('target_conf').disabled = true;
        document.getElementById('grayscale').disabled = true;
        document.getElementById('rgb').disabled = true;
        document.getElementById('init').disabled = true;
        document.getElementById('max_queries').disabled = true;
        document.getElementById('start_button').disabled = true;
    }
    
    function start() {
        let target_class_element = document.getElementById('target_class');
        let target_conf_element = document.getElementById('target_conf');
        let rgb_element = document.getElementById('rgb');
        let init_element = document.getElementById('init');
        let max_queries_element = document.getElementById('max_queries');

        let target_class = target_class_element[target_class_element.selectedIndex].value;
        let target_conf = target_conf_element.value;
        let rgb = rgb_element.checked;
        let init = init_element.checked;
        let max_queries = max_queries_element.value;

        showSpinner();

        socket.send(JSON.stringify({
            start: true,
            target_class: target_class,
            target_conf: target_conf,
            init: init,
            max_queries: max_queries,
            rgb: rgb
        }));
    }

    function reset() {
        let div = document.getElementById('images');
        let final_image = document.getElementById('final_image');
        let opt = document.getElementById('target_class');

        disableForm(false);

        document.getElementById("image_file").value = "";
        document.getElementById('messages').value = '';

        while (div.firstChild) {
            div.removeChild(div.firstChild);
        }

        while (opt.firstChild) {
            opt.removeChild(opt.firstChild);
        }

        while (final_image.firstChild) {
            final_image.removeChild(div.firstChild);
        }
    }
    
    function foolingFinished(src) {
        let img = document.createElement("img");
        img.src = src;
        img.classList.add("img-thumbnail");

        let div = document.getElementById('final_image');
        div.appendChild(img);

        hideSpinner();
    }

    function sendImage(element) {
        let file = element.files[0];
        let reader = new FileReader();

        reader.onloadend = function () {
            showSpinner();
            socket.send(JSON.stringify({
                'image' : reader.result
            }))
        };
        reader.readAsDataURL(file);
    }

    function displayMessage(message) {
        let textarea = document.getElementById('messages');
        textarea.value += '\n';
        textarea.value += message;
    }
    
    function startSocket() {
         socket = new WebSocket("ws://localhost:8008/");

        socket.addEventListener('open', function (event) {
            socket.send(JSON.stringify({
                id: uuidv4()
            }))
        });

        socket.addEventListener('message', function (event) {

            let reader = new FileReader();
            reader.onload = function() {

                let msg = JSON.parse(reader.result);

                if(msg.hasOwnProperty('labels')) {
                    hideSpinner();
                    fillDropdown(msg.labels)
                }

                if(msg.hasOwnProperty('data')) {
                    console.log('DATA:', msg.data);
                    displayMessage(msg.data)
                }

                if(msg.hasOwnProperty('done') && msg.hasOwnProperty('image')){
                    if(msg.done) {
                        hideSpinner();
                        foolingFinished(msg.image);
                    } else {
                        appendImages(msg.image)
                    }
                }

            };
            reader.readAsText(event.data);
        });

        socket.addEventListener('error', function (event) {
            socket.close();
            $('#myModal').modal('show');
        });

        socket.addEventListener('close', function (event) {
            $('#myModal').modal('show');
        });
    }
