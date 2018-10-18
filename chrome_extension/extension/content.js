	function factchecking(claim, cb) {
		var xhr = new XMLHttpRequest();
		xhr.open('GET', "https://0.0.0.0:5000/api/evidence?claim="+encodeURIComponent(claim));
		xhr.send();
		xhr.onreadystatechange = () => {
			if (xhr.readyState == 4 && xhr.status == 200) {
				var r = JSON.parse(xhr.responseText);
				console.log(r);
        cb(r);
			}
		}
	}
	
	function uniq(a) {
    return a.sort((e)=> e.sourceItemOriginFeedName).filter(function(item, pos, ary) {
        return !pos || item.sourceItemOriginFeedName != ary[pos - 1].sourceItemOriginFeedName;
    })
}

			console.log("ext ok");

var parentContainerId = "MainW" //"ob-read-more-selector"
		
	if(!window.CurrentSelection){
		CurrentSelection = {}
	}
	
	CurrentSelection.Selector = {}
	
	//get the current selection
	CurrentSelection.Selector.getSelected = function(){
		var sel = '';
		if(window.getSelection){
			sel = window.getSelection()
		}
		else if(document.getSelection){
			sel = document.getSelection()
		}
		else if(document.selection){
			sel = document.selection.createRange()
		}
		return sel
	}
	//function to be called on mouseup
	CurrentSelection.Selector.mouseup = function(){
		
		var st = CurrentSelection.Selector.getSelected()
		if(document.selection && !window.getSelection){
			var range = st
			range.pasteHTML("<span class='selectedText'>" + range.htmlText + "</span>");			
		}
		else{
			var range = st.getRangeAt(0);
			var newNode = document.createElement("span");
			newNode.setAttribute("class", "selectedText");
			range.surroundContents(newNode);
			//
		   var getTitle = newNode.innerHTML;
		   newNode.setAttribute("title", getTitle);

		   //
		   var popDiv = document.createElement('span');
		   popDiv.setAttribute('class', 'popDiv');
       
		    factchecking(getTitle, (r) => {
			let all = r["evidences"].map((e)=>{ 
				let o = {
					"sourceItemOriginFeedName": e["document"]["header"]["sourceItemOriginFeedName"],
					"label": e["group_fact_check"],
					"url": e["document"]["header"]["sourceItemIdAtOrigin"],
					"evidence":e["element"]["text"]
				}
				return o;
			}) 
			
			all_unique = []
			for(j = 0; j < all.length; j += 1){
					failed = false;
					for(k = 0; k < all_unique.length; k += 1){
						if (all[j].sourceItemOriginFeedName == all_unique[k].sourceItemOriginFeedName){
							failed = true;
						}
					}
					if(!failed){
						all_unique.push(all[j]);
					}
			}
			
			console.log(all);
			support_ev = all_unique.filter(e => e["label"] == "SUPPORTS");
			reject_ev = all_unique.filter(e => e["label"] == "REJECT");
			other_ev = all_unique.filter(e => e["label"] == "NOT ENOUGH INFO");
			
			
			
			
			
			let line = "";
			for (i = 0; i < support_ev.length; i += 1){	
				line += "<li>" + "<a href=\""+support_ev[i].url +"\" target=\"_blank\"\>" + support_ev[i].sourceItemOriginFeedName  + "</a>" + " | <color_support>" + support_ev[i].label + "</color_support><ul><li><i>" + support_ev[i].evidence + "</i></ul></li></li>";
			}
			
			for (i = 0; i < reject_ev.length; i += 1){	
				line += "<li>" + "<a href=\""+reject_ev[i].url +"\" target=\"_blank\"\>" + reject_ev[i].sourceItemOriginFeedName  + "</a>" + " | <color_reject>" + reject_ev[i].label + "</color_reject><ul><li><i>" + reject_ev[i].evidence + "</i></ul></li></li>";
			}
			
			for (i = 0; i < other_ev.length; i += 1){	
				line += "<li>" + "<a href=\""+other_ev[i].url +"\" target=\"_blank\"\>" + other_ev[i].sourceItemOriginFeedName  + "</a>" + " | <color_other>" + other_ev[i].label + "</color_other><ul><li><i>" + other_ev[i].evidence + "</i></ul></li></li>";
			}
			
        	popDiv.innerHTML = "Overall Result: " +  "<color_"+r["global_fc"].replace(" ", "_") + ">" +r["global_fc"]    +   "</color_"+r["global_fc"].replace(" ", "_") + ">" + ": <ul>" + line + "</ul>";
			
			//+ " </ul>\nRefutes: " + reject_ev.map(x=> x.sourceItemOriginFeedName + " | " + x.label).join(" ")+ "\nNot enough info: " + other_ev.map(x=> x.sourceItemOriginFeedName + " | " + x.label).join(" ");
        });

		   if(newNode.innerHTML.length > 0) {
		    newNode.appendChild(popDiv);
		   }		   
		   //Remove Selection: To avoid extra text selection in IE  
		   if (window.getSelection) {
		     window.getSelection().removeAllRanges();
		   }
	       else if (document.selection){ 
	        document.selection.empty();
	       }
	       //
		}
	}
        
	$(function(){

		$("#"+parentContainerId).on('mouseup', function(){
			console.log("mouseup ok");
			 $('span.selectedText').contents().unwrap();
			// $(this).find('span.popDiv').remove();			
		});

		$("#"+parentContainerId).bind("mouseup",CurrentSelection.Selector.mouseup);	
	})
