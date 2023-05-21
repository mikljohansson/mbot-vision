#ifndef _MV_HTTPD_INDEX_H_
#define _MV_HTTPD_INDEX_H_

#include <Arduino.h>

String indexDocument = R"doc(<html>
  <head>
    <link rel="icon" href="data:;base64,=">
    <script type="text/javascript">
        const post = (url, onload) => {
            var xmlHttp = new XMLHttpRequest();

            if (onload) {
              xmlHttp.onload = () => onload(xmlHttp.response);
            }

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

        const handleFlashToggle = () => {
          const flash = document.getElementById("flash");
          flash.value = flash.value > 0 ? 0 : 125;
        };

        let source = null;
        
        const connectEvents = () => {
          source = new EventSource('/events');
          source.onmessage = (e) => {
              const crosshair = document.getElementById("crosshair");
              const blob = JSON.parse(e.data);
              const rect = crosshair.parentElement.getBoundingClientRect();
              crosshair.style.left = Math.round(rect.width * blob.x) + "px";
              crosshair.style.top = Math.round(rect.height * blob.y) + "px";
              crosshair.style.display = blob.detected ? "inline" : "none";
              //crosshair.style.color = blob.color;
              //console.log(e.data);
          };
        }

        const handleToggleDetectorStream = () => {
          post("/toggleDetectorStream");
        };

        const handleToggleImageLogging = () => {
          post("/toggleImageLogging", (response) => {
            console.log("Image logger is now", response);
            document.getElementById("imageLoggingButton").style.color = response === "active" ? "#ff0000" : "#999";
          });
        };

        const handleLoad = () => {
          connectEvents();
        };
    </script>
    <style>
      .body {
        background-color: #000;
        display: flex;
        justify-content: center;
      }

      #container {
        display: inline-block;
        position: relative;
        width: 100%;
        height: 100%;
        max-width: 640px;
        max-height: 480px;
        margin: 5px;
        background-image: url('/stream');
        background-repeat: no-repeat;
        background-size: contain;
        background-position: top center;
        image-rendering: pixelated;
      }

      #crosshair {
        position: absolute;
        color: #999;
        font-size: 500%;
        transform: translate(-50%, -50%);
        display: none;
      }

      .container td {
        vertical-align: top;
      }

      .inputcell {
        width: 99%;
      }

      .inputcell input {
        width: 100%;
      }

      .button {
        cursor: pointer;
        font-size: 150%;
      }
    </style>
  </head>
  <body class="body" onLoad="handleLoad();">
    <div id="container">
      <div id="crosshair">&#x2316</div>
      <table>
        <td class="button" onClick="javascript:handleFlashToggle();" title="Toggle the floodlights on/off">&#x1F4A1;</td>
        <td class="inputcell"><input type="range" min="0" max="175" value="0" id="flash" oninput="handleFlash(this.value);" onchange="handleFlash(this.value);" title="Floodlight brightness"></td>
        <td class="button" style="color: #999;" onClick="javascript:handleToggleDetectorStream();" title="Show ML model input and output">&#x267B;</td>
        <td id="imageLoggingButton" class="button" style="color: #999;" onClick="javascript:handleToggleImageLogging();" title="Record training data to SDcard">&#x25C9;</td>
      </table>
    </div>
    <div class="container detector" />
  <body>
<html>)doc";

#endif