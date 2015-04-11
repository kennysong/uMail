require File.dirname(__FILE__) + '/../test_helper'
require 'thread_controller'

# Re-raise errors caught by the controller.
class ThreadController; def rescue_action(e) raise e end; end

class ThreadControllerTest < Test::Unit::TestCase
  fixtures :email_threads

  def setup
    @controller = ThreadController.new
    @request    = ActionController::TestRequest.new
    @response   = ActionController::TestResponse.new
  end

  def test_index
    #get :index
    #assert_response :success
    #assert_template 'list'
  end

  def test_list
    #get :list

    #assert_response :success
    #assert_template 'list'

    #assert_not_nil assigns(:email_threads)
  end

  def test_show
    #get :show, :id => 1

    #assert_response :success
    #assert_template 'show'

    #assert_not_nil assigns(:email_thread)
    #assert assigns(:email_thread).valid?
  end

  def test_new
    #get :new

    #assert_response :success
    #assert_template 'new'

    #assert_not_nil assigns(:email_thread)
  end

  def test_create
    num_email_threads = EmailThread.count

    #post :create, :email_thread => {}

    #assert_response :redirect
    #assert_redirected_to :action => 'list'

    #assert_equal num_email_threads + 1, EmailThread.count
  end

  def test_edit
    #get :edit, :id => 1

    #assert_response :success
    #assert_template 'edit'

    #assert_not_nil assigns(:email_thread)
    #assert assigns(:email_thread).valid?
  end

  def test_update
    #post :update, :id => 1
    #assert_response :redirect
    #assert_redirected_to :action => 'show', :id => 1
  end

  def test_destroy
    #assert_not_nil EmailThread.find(1)

    #post :destroy, :id => 1
    #assert_response :redirect
    #assert_redirected_to :action => 'list'

    #assert_raise(ActiveRecord::RecordNotFound) {
    #  EmailThread.find(1)
    #}
  end
end
