#ifndef _MV_HTTPD_INDEX_H_
#define _MV_HTTPD_INDEX_H_

#include <Arduino.h>

String indexDocument = R"doc(<html>
  <head>
    <link rel="icon" href="data:;base64,=">
    <script type="text/javascript">
        const post = (url) => {
            var xmlHttp = new XMLHttpRequest();
            xmlHttp.open("POST", url, true);
            xmlHttp.send(null);
        };
        
        const debounce = (callback, wait = 250) => {
            let timeout = null;
            let useargs = null;

            return (...args) => {
                if (!timeout) {
                    useargs = args;
                    timeout = setTimeout(() => {
                        timeout = null;
                        callback(...useargs);
                    }, wait);
                }
                else {
                    useargs = args;
                }
            };
        };

        const handleFlash = debounce((value) => {
            post(`/flash?v=${value}`);
        });

        const source = new EventSource('/events');
        source.onmessage = (e) => {
            const crosshair = document.getElementById("crosshair");
            const blob = JSON.parse(e.data);
            const rect = crosshair.parentElement.getBoundingClientRect();
            crosshair.style.left = Math.round(rect.width * blob.x) + "px";
            crosshair.style.top = Math.round(rect.height * blob.y) + "px";

        };
    </script>
    <style>
      .body {
        background-color: #000;
        display: flex;
        justify-content: center;
      }

      .container {
        display: inline-block;
        position: relative;
        width: 100%;
        height: 100%;
        max-width: 640px;
        max-height: 480px;
        padding: 5px;
        background-image: url("/stream");
        background-repeat: no-repeat;
        background-size: contain;
        background-position: top center;
      }

      #crosshair {
        position: absolute;
        color: white;
        font-size: 200%;
        transform: translate(-50%, -50%);
      }

      .container td {
        vertical-align: top;
      }

      .inputcell {
        width: 99%;
      }

      .container input {
        width: 100%;
      }
    </style>
  </head>
  <body class="body">
    <div class="container">
      <div id="crosshair">&#x2316</div>
      <table>
        <td>&#x1F4A1;</td>
        <td class="inputcell"><input type="range" min="0" max="175" value="0" id="flash" oninput="handleFlash(this.value);" onchange="handleFlash(this.value);"></td>
      </table>
    </div>
  <body>
<html>)doc";

#endif