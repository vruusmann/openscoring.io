/**
 * Theme functions file
 *
 * Contains handlers for navigation, accessibility, header sizing
 * footer widgets and Featured Content slider
 *
 */
( function( $ ) {
    $(document).ready(function() {
        $('#menu-toggle').click(function(e){
            $('body, #menu-toggle, .c-head__nav').toggleClass('toggled');
            e.preventDefault();
        });
        var header = $('.c-head'),
            fixed = 'c-head--fixed',
            shadow = 'c-head--shadow';
        function scroll() {
            window.scrollY >= 35 && header.addClass(fixed) || header.removeClass(fixed),
            window.scrollY >= 70 && header.addClass(shadow) || header.removeClass(shadow);
            $('.c-head__menu a').each(function() {

            });
        }
        $(window).on('load resize scroll', scroll);
    });

    $.fn.initSwiper = function() {
        var id = $(this).attr('id');
        new Swiper('#'+id, {
            slidesPerView: 1,
            speed: 1000,
            loop: true,
            loopedSlides: 4,
            observer: true,
            observeParents: true,
            autoplay: {
                delay: 5000
            },
            navigation: {
                nextEl: '#'+id+'n',
                prevEl: '#'+id+'p'
            }
        });
    }
    $('.p-txtmedia__media-slider').each(function () {
        $(this).initSwiper();
    });
    /*
    function offsetTop(el) {
        var rect = el.getBoundingClientRect(),
            scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        return rect.top + scrollTop;
    }
     var $root = $('html, body');
     function smoothScrollingTo(target, event){
        if (!target) return;
        event.preventDefault();
        $root.animate({scrollTop:offsetTop(document.querySelector(target))}, 800);
        setTimeout(function() {
            window.location.hash = target
        }, 800)
    }
    $('a:not(.nav-link)[href*=\\#]').on('click', function(event){
        if (this.pathname !== window.location.pathname) return;
        if (this.hash.length > 0) {
            smoothScrollingTo(this.hash, event);
        } else {
            event.preventDefault();
            $root.animate({scrollTop:0}, 800);
        }

        $('body, #menu-toggle, .c-head__nav').removeClass('toggled');
    });
    $(document).ready(smoothScrollingTo(location.hash)); */

    /* $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
        $('code').SimpleBar('recalculate');
    }); */

    $('button.copy-code').on('click', function(e) {
        var t = $(this);
        console.log(t.data('target'));
        console.log($(t.data('target')).select());
        $(t.data('target')).select();
        document.execCommand('copy');
        t.tooltip({placement: 'right', trigger: 'manual', title: 'Copied'});
        t.tooltip('show');
        setTimeout(function(){t.tooltip('hide')}, 1500)
        e.preventDefault();
    });

})( jQuery );